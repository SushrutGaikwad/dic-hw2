from pyspark import SparkContext
from pyspark.streaming import StreamingContext


def generate_kmers(line: str, k: int = 3):
    """
    Generate all k-mers of length `k` from a line of text.
    """
    line = line.strip().replace(" ", "").lower()
    return [line[i : i + k] for i in range(len(line) - k + 1)]


if __name__ == "__main__":
    # Create SparkContext and StreamingContext
    sc = SparkContext("local[2]", "KMerCount")
    ssc = StreamingContext(sc, 10)  # 10-second batches

    # Create a DStream from TCP source
    lines = ssc.socketTextStream("localhost", 9999)

    # Generate k-mers and count them
    kmers = lines.flatMap(lambda line: generate_kmers(line, k=3))
    kmer_counts = kmers.map(lambda kmer: (kmer, 1)).reduceByKey(lambda a, b: a + b)

    # Print the results
    kmer_counts.pprint()

    # Start streaming
    ssc.start()
    ssc.awaitTermination()
