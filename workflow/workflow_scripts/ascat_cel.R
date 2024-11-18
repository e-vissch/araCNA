#run ASCAT functions
library(optparse)
library(ASCAT)
options(bitmapType='cairo-png')

option_list = list(
  make_option(c("-c", "--case_id"), type="character", default=NULL, help="Case ID", metavar="character"),
  make_option(c("-t", "--lrr_baf_file_tumor"), type="character", default=NULL, help="Path to the lrr file", metavar="FILE"),
  make_option(c("-n", "--lrr_baf_file_normal"), type="character", default=NULL, help="Path to the lrr file", metavar="FILE"),
  make_option(c("-s", "--snp_pos_file"), type="character", default=NULL, help="Path to SNP pos file", metavar="FILE"),
  make_option(c("--snp_gc_dir"), type="character", default=NULL, help="SNP GC dir", metavar="FILE"),
  make_option(c("--birdseed_dir"), type="character", default=NULL, help="birdseed_dir", metavar="FILE"),
  make_option(c("-o", "--out_dir"), type="character", default=NULL, help="Path to the output directory", metavar="FILE")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)


# Create the directory if it doesn't exist
if (!dir.exists(opt$out_dir)) {
  dir.create(opt$out_dir, recursive = TRUE)
}

lrrbaf_t = read.table(opt$lrr_baf_file_tumor, header = T, sep = "\t", row.names=1)
lrrbaf_n = read.table(opt$lrr_baf_file_normal, header = T, sep = "\t", row.names=1)

SNPpos = read.table(opt$snp_pos_file,header=T,sep="\t",row.names=1)
SNPpos <- SNPpos[, 1:2]
tumor_sample = sub(".CEL.Log.R.Ratio","",colnames(lrrbaf_t)[3])
normal_sample = sub(".CEL.Log.R.Ratio","",colnames(lrrbaf_n)[3])

Tumor_LogR = lrrbaf_t[rownames(SNPpos),3,drop=F]
colnames(Tumor_LogR) = opt$case_id

Tumor_BAF = lrrbaf_t[rownames(SNPpos),4,drop=F]
colnames(Tumor_BAF) = opt$case_id

Normal_LogR = lrrbaf_n[rownames(SNPpos),3,drop=F]
colnames(Normal_LogR) = opt$case_id

Normal_BAF = lrrbaf_n[rownames(SNPpos),4,drop=F]
colnames(Normal_BAF) = opt$case_id

#replace 2's by NA
Tumor_BAF[Tumor_BAF==2]=NA
Normal_BAF[Normal_BAF==2]=NA

# Tumor_LogR: correct difference between copy number only probes and other probes
CNprobes = substring(rownames(SNPpos),1,2)=="CN"

Tumor_LogR[CNprobes,1] = Tumor_LogR[CNprobes,1]-mean(Tumor_LogR[CNprobes,1],na.rm=T)
Tumor_LogR[!CNprobes,1] = Tumor_LogR[!CNprobes,1]-mean(Tumor_LogR[!CNprobes,1],na.rm=T)

Normal_LogR[CNprobes,1] = Normal_LogR[CNprobes,1]-mean(Normal_LogR[CNprobes,1],na.rm=T)
Normal_LogR[!CNprobes,1] = Normal_LogR[!CNprobes,1]-mean(Normal_LogR[!CNprobes,1],na.rm=T)

# limit the number of digits:
Tumor_LogR = round(Tumor_LogR,4)
Normal_LogR = round(Normal_LogR,4)

out_stub = paste0(opt$out_dir, "/", opt$case_id)

file.tumor.BAF <- paste(out_stub, ".tumor.BAF.txt", sep="")
file.tumor.LogR <- paste(out_stub, ".tumor.LogR.txt", sep="")
file.normal.BAF <- paste(out_stub, ".normal.BAF.txt", sep="")
file.normal.LogR <- paste(out_stub, ".normal.LogR.txt", sep="")


write.table(cbind(SNPpos,Tumor_BAF),file.tumor.BAF,sep="\t",row.names=T,col.names=NA,quote=F)
write.table(cbind(SNPpos,Normal_BAF),file.normal.BAF,sep="\t",row.names=T,col.names=NA,quote=F)

write.table(cbind(SNPpos,Tumor_LogR),file.tumor.LogR,sep="\t",row.names=T,col.names=NA,quote=F)
write.table(cbind(SNPpos,Normal_LogR),file.normal.LogR,sep="\t",row.names=T,col.names=NA,quote=F)


gender <- read.table(paste0(opt$birdseed_dir, "/birdseed.report.txt"), sep="\t", skip=66, header=T)
sex <- as.vector(gender[,"computed_gender"])
sex[sex == "female"] <- "XX"
sex[sex == "male"] <- "XY"
sex[sex == "unknown"] <- "XX"

# expects list
ascat.bc <- ascat.loadData(file.tumor.LogR, file.tumor.BAF, file.normal.LogR, file.normal.BAF, chrs=c(1:22, "X"), gender=sex)


print(paste0(opt$snp_gc_dir, "/GC_AffySNP6_102015.txt"))
#GC correction for SNP6 data
ascat.bc = ascat.correctLogR(ascat.bc, paste0(opt$snp_gc_dir, "/GC_AffySNP6_102015.txt"))

ascat.plotRawData(ascat.bc, img.dir=opt$out_dir)

ascat.bc <- ascat.aspcf(ascat.bc, out.dir=opt$out_dir)

ascat.plotSegmentedData(ascat.bc, img.dir=opt$out_dir)

ascat.output <- ascat.runAscat(ascat.bc, img.dir=opt$out_dir)

#save ASCAT results
write.table(ascat.output$segments, file=paste(out_stub,".segments.txt",sep=""), sep="\t", quote=F, row.names=F)
write.table(ascat.output$aberrantcellfraction, file=paste(out_stub,".acf.txt",sep=""), sep="\t", quote=F, row.names=F)
write.table(ascat.output$ploidy, file=paste(out_stub,".ploidy.txt",sep=""), sep="\t", quote=F, row.names=F)

save.image(paste(out_stub, opt$case_id,".RData",sep=""))
