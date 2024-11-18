# Below commands are for creating segmentation files and producing some QC images
# opt$out_dir used to construct names for files produced by this script

library(HMMcopy)
library(optparse)
options(bitmapType='cairo-png')

option_list <- list(
    make_option(c("--normal_reads"), type = "character", default = NULL,
                metavar = "FILE"),
    make_option(c("--tumor_reads"), type = "character", default = NULL,
                metavar = "FILE"),
    make_option(c("--gc_content"), type = "character", default = NULL,
                metavar = "FILE"),
    make_option(c("--ref_mappable"), type = "character", default = NULL,
                metavar = "FILE"),
    make_option(c("-o", "--out_dir"), type="character", default=NULL, help="Path to the output directory", metavar="character")
)

# Parse command-line options
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create the directory if it doesn't exist
if (!dir.exists(opt$out_dir)) {
  dir.create(opt$out_dir, recursive = TRUE)
}



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

preprocess_timing <- system.time({
tum_uncorrected_reads <- wigsToRangedData(opt$tumor_reads, opt$gc_content, opt$ref_mappable)  
norm_uncorrected_reads <- wigsToRangedData( opt$normal_reads, opt$gc_content, opt$ref_mappable)
tum_corrected_copy <- correctReadcount(tum_uncorrected_reads)
norm_corrected_copy <- correctReadcount(norm_uncorrected_reads)

# Should take no longer than a few minutes on a human genome.
# The correctReadcount requires at least about 1000 bins to work properly.
# 2. Segmentation

# Below commands in R
# Normalizing Tumour by Normal
tum_corrected_copy$copy <- tum_corrected_copy$copy - norm_corrected_copy$copy
})

timing_results <- append_time(preprocess_timing, "preprocess", timing_results)


main_timing <- system.time({
param <- HMMsegment(tum_corrected_copy, getparam = TRUE) # retrieve converged parameters via EM
param$mu <- log(c(1, 1.4, 2, 2.7, 3, 4.5) / 2, 2)
param$m <- param$mu

segmented_copy <- HMMsegment(tum_corrected_copy, param) # perform segmentation via Viterbi
}) 
timing_results <- append_time(main_timing, "main", timing_results)


# 3. Export
# Export to SEG format for CNAseq segmentation
post_process_timing <- system.time({
segFile<-paste(opt$out_dir, "seg", sep = ".")
tsvFile<-paste(opt$out_dir, "tsv", sep = ".")

rangedDataToSeg(tum_corrected_copy, file = segFile)
write.table(segmented_copy$segs, file = tsvFile, quote = FALSE, sep = "\t")

options(bitmapType="cairo")
# 4. Visualization - produce some images with hard-coded dimensions

# Bias plots:
print("Producing CG Bias plot...")
png(filename = paste(opt$out_dir,"bias_plot","png", sep="."),width = 1200, height = 580, units="px", pointsize=15, bg="white")
plotBias(tum_corrected_copy)  # May be one plot per comparison  1200x580
dev.off()

chroms<-unique(segmented_copy$segs$chr)

# Segmentation plots:
# need to do it one plot per chromosome 1200x450
print("Producing Segmentation plots...")
par(mfrow = c(1, 1))
for (c in 1:length(chroms)) {
 if (!grepl("_",chroms[c]) && !grepl("M",chroms[c])) {
	 png(filename = paste(opt$out_dir,"s_plot", chroms[c], "png", sep="."),width = 1200, height = 450, units="px", pointsize=15, bg="white")
	 plotSegments(tum_corrected_copy, segmented_copy, pch = ".", ylab = "Tumour Copy Number", xlab = "Chromosome Position",chr = chroms[c], main = paste("Segmentation for Chromosome",chroms[c], sep=" "))
	 cols <- stateCols()
	 legend("topleft", c("HOMD", "HETD", "NEUT", "GAIN", "AMPL", "HLAMP"), fill = cols, horiz = TRUE, bty = "n", cex = 0.9)
	 dev.off()
 } else {
	 print(paste("Chromosome",c,"cannot be plotted with  plotSegments",sep = " "))
 }
}
})

timing_results <- append_time(post_process_timing, "post_process", timing_results)
write.csv(timing_results, file = paste0(opt$out_dir, "/timings.csv"), row.names = FALSE)