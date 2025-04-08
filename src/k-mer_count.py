from pyspark import SparkContext
from pyspark.streaming import StreamingContext

def generate_kmers(line: str, k: int = 3):
    """
    Generate all k-mers of length k from a line of text using a for-loop.
    Example: "abcdef", k=3 → ['abc', 'bcd', 'cde', 'def']
    """
    kmers = []
    line = line.strip().lower().replace(" ", "")
    line_length = len(line)

    if line_length >= k:
        for i in range(line_length - k + 1):
            kmer = line[i:i+k]
            kmers.append(kmer)

    return kmers

sc = SparkContext("local[2]", "KMerStreaming")
ssc = StreamingContext(sc, batchDuration=10)

lines = ssc.socketTextStream("localhost", 9999)

kmers = lines.flatMap(lambda line: generate_kmers(line, k=3))

kmer_counts = kmers.map(lambda kmer: (kmer, 1)).reduceByKey(lambda a, b: a + b)

kmer_counts.foreachRDD(
    lambda rdd: print(
        "\nK-mer Counts:\n" + "\n".join([f"{kmer}: {count}" for kmer, count in rdd.collect()])
    )
)

ssc.start()
ssc.awaitTermination()