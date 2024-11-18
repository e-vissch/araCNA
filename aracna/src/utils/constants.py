MODULE_DEFUALT_LOSS = "default_loss"
SEQLEN_REMOVE_KEYS = ["seqlen_warmup", "global_batch_size"]
MODEL_AFFECTED_TASK_INFO_KEYS = [
    "supervised_predict_keys",
    "max_seq_length",
    "max_tot_cn_arch",
    "maj_out_dim" "min_out_dim",
]

READ_KEY = "read_depth"
PURITY_KEY = "purity"
SEQ_TOKEN = 0
GLOBAL_TOKEN = 1

READ_IN_DIM = 0
BAF_IN_DIM = 1
TOKEN_IN_DIM = 2
READ_GLOB_OUT_DIM = 0
MAX_METRICS = ["discrete_accuracy"]

# Data constants, maybe need new file?
N_CHROM = 23  # 1-22, X (no Y)
AVG_SNP_DIST = 1.7e3
# In MBP, rough estimate of chrom sizes
CHROM_SIZE_DICT = {
    1: 248,
    2: 242,
    3: 198,
    4: 190,
    5: 180,
    6: 171,
    7: 159,
    8: 146,
    9: 141,
    10: 135,
    11: 135,
    12: 133,
    13: 115,
    14: 107,
    15: 102,
    16: 90,
    17: 81,
    18: 78,
    19: 63,
    20: 63,
    21: 48,
    22: 50,
    23: 156,
}
