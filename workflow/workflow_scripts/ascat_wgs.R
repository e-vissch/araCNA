library(ASCAT)
library(optparse)
options(bitmapType='cairo-png')

# Define command-line options
option_list <- list(
    make_option(c("--tumor_bam"), type = "character", default = NULL,
                metavar = "FILE"),
    make_option(c("--normal_bam"), type = "character", default = NULL,
                metavar = "FILE"),
    make_option(c("-t", "--tumor_name"), type = "character", default = NULL,
                metavar = "STRING"),
    make_option(c("-n", "--normal_name"), type = "character", default = NULL,
                metavar = "STRING"),
    make_option(c("-d", "--ascat_data_dir"), type = "character", default = NULL,
                metavar = "STRING"),
    make_option(c("-c", "--include_correction"), type = "character", default = TRUE,
                metavar = "BOOL"),
    make_option(c("--alleleCounter_exe"), type="character", default=NULL, metavar="FILE"),
    make_option(c("--nthreads"), type="character", default=8, metavar="NUM"),
    make_option(c("-o", "--out_dir"), type="character", default=NULL, help="Path to the output directory", metavar="character")
)


# Parse command-line options
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create the directory if it doesn't exist
if (!dir.exists(opt$out_dir)) {
  dir.create(opt$out_dir, recursive = TRUE)
}

setwd(opt$out_dir)

out_stub = paste0(opt$out_dir, "/")

# Use the command-line arguments
params.tumor_logr = paste0(out_stub, "tumor_LogR.txt")
params.tumor_baf = paste0(out_stub, "tumor_BAF.txt")
params.normal_logr = paste0(out_stub, "normal_LogR.txt")
params.normal_baf = paste0(out_stub, "normal_BAF.txt")
params.gender = "XX"
params.genomeVersion = "hg38"

# Initialize an empty data frame to store the results
timing_results <- data.frame(
  Operation = character(),
  User_Time_Seconds = numeric(),
  System_Time_Seconds = numeric(),
  Elapsed_Time_Seconds = numeric(),
  stringsAsFactors = FALSE
)

append_time <- function(timing_result, operation_name, results_df) {
  # Store the timing results and result in the data frame
  results_df <- rbind(results_df, data.frame(
    Operation = operation_name,
    User_Time_Seconds = timing_result["user.self"],
    System_Time_Seconds = timing_result["sys.self"],
    Elapsed_Time_Seconds = timing_result["elapsed"]
  ))
  return(results_df)
}

preprocess_timing <- system.time(ascat.prepareHTS(
  tumourseqfile = opt$tumor_bam,
  normalseqfile = opt$normal_bam,
  tumourname = opt$tumor_name,
  normalname = opt$normal_name,
  allelecounter_exe = opt$alleleCounter_exe,
  alleles.prefix = paste0(opt$ascat_data_dir, "/G1000_alleles_hg38/G1000_alleles_hg38_chr"),
  loci.prefix = paste0(opt$ascat_data_dir, "/G1000_loci_hg38/G1000_loci_hg38_chr"),
  gender = params.gender,
  genomeVersion = params.genomeVersion,
  nthreads = opt$nthreads,
  tumourLogR_file = params.tumor_logr,
  tumourBAF_file = params.tumor_baf,
  normalLogR_file = params.normal_logr,
  normalBAF_file = params.normal_baf)
)
print(preprocess_timing)
timing_results <- append_time(preprocess_timing, "preprocess", timing_results)

ascat.bc = ascat.loadData(Tumor_LogR_file = params.tumor_logr, Tumor_BAF_file = params.tumor_baf, Germline_LogR_file = params.normal_logr, Germline_BAF_file = params.normal_baf, gender = params.gender, genomeVersion = params.genomeVersion)


img_prefix="Raw_"
if (opt$include_correction) {
  ascat.plotRawData(ascat.bc, img.prefix = "Before_correction_", img.dir=opt$out_dir)
  gc_file=paste0(opt$ascat_data_dir,"/GC_G1000_hg38.txt")
  rt_file=paste0(opt$ascat_data_dir,"/RT_G1000_hg38.txt")
  ascat.bc = ascat.correctLogR(ascat.bc, GCcontentfile = gc_file, replictimingfile = rt_file)
  img_prefix="After_correction_"
}

ascat.plotRawData(ascat.bc, img.prefix = img_prefix, img.dir=opt$out_dir)

# List of penalty terms to loop over
penalty_terms <- c(35, 50, 70)

# Loop over each penalty term
for (penalty in penalty_terms) {
  new_timing <- timing_results
  total_timing <- system.time({
  # Define output directory for the current penalty term
  out_dir <- paste0(opt$out_dir, "penalty_", penalty)  
  # Ensure the directory exists
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  # Run ascat.aspcf with the current penalty term

  aspcf_timing <- system.time({
  ascat.bc <- ascat.aspcf(ascat.bc, penalty = penalty, out.dir = out_dir)
  })
  new_timing <- append_time(aspcf_timing, "aspcf", new_timing)
  # Plot segmented data
  ascat.plotSegmentedData(ascat.bc, img.dir = out_dir)
  # Run ascat.runAscat

  run_ascat_time <- system.time({
  ascat.output <- ascat.runAscat(ascat.bc, gamma = 1, write_segments = TRUE, img.dir = out_dir)
  })
  new_timing <- append_time(run_ascat_time, "main_ascat", new_timing)
  # Get metrics
  QC <- ascat.metrics(ascat.bc, ascat.output)
  # Save QC metrics and objects
  write.csv(QC, file = paste0(out_dir, "/qc.csv"), row.names = FALSE)
  save(ascat.bc, ascat.output, QC, file = paste0(out_dir, "/ASCAT_objects.Rdata"))
  }) 
  new_timing <- append_time(total_timing, "main_ascat", new_timing)
  write.csv(timing_results, file = paste0(out_dir, "/timings.csv"), row.names = FALSE)
}
