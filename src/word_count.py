from pyspark import SparkContext
import string
import shutil
from pathlib import Path
from typing import Set


def clean_word(word: str) -> str:
    """
    Remove punctuation and convert word to lowercase.
    """
    return word.translate(str.maketrans("", "", string.punctuation)).lower()


def save_output(rdd, output_path: str) -> None:
    """
    Save RDD to output path. If the directory exists, delete it first.
    """
    path = Path(output_path)
    if path.exists() and path.is_dir():
        print(f"[INFO] Output path '{output_path}' exists. Deleting it first.")
        shutil.rmtree(path)
    rdd.saveAsTextFile(output_path)
    print(f"[INFO] Saved RDD to '{output_path}'")


def print_top_n(rdd, n: int = 25) -> None:
    """
    Print the top N elements from the RDD.
    """
    top_items = rdd.take(n)
    for word, count in top_items:
        print(f"{word}: {count}")


if __name__ == "__main__":
    # Initialize SparkContext
    sc = SparkContext("local", "WordCount")

    # Define custom stop words
    stop_words: Set[str] = set(
        [
            "the",
            "and",
            "a",
            "an",
            "to",
            "in",
            "is",
            "it",
            "of",
            "that",
            "this",
            "on",
            "was",
            "with",
            "as",
            "for",
            "but",
            "by",
            "be",
            "at",
            "are",
            "or",
            "he",
            "she",
            "i",
            "you",
            "they",
            "we",
            "his",
            "her",
            "their",
            "my",
            "me",
            "your",
            "has",
            "have",
            "had",
            "will",
            "would",
            "can",
            "could",
            "should",
            "do",
            "does",
            "did",
        ]
    )

    # Read input files
    book1 = sc.textFile("book1.txt")
    book2 = sc.textFile("book2.txt")
    full_text = book1.union(book2)

    # Task 1.1 - Basic Word Count
    basic_counts = (
        full_text.flatMap(lambda line: line.split())
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
    )
    save_output(basic_counts, "output_1.txt")

    # Task 1.2 - Extended Word Count
    extended_counts = (
        full_text.flatMap(lambda line: line.split())
        .map(lambda word: clean_word(word))
        .filter(lambda word: word and word not in stop_words)
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda x: x[1], ascending=False)
    )
    save_output(extended_counts, "output_1_extended.txt")

    # Task 1.4 - Top 25 from book1.txt only (extended)
    book1_extended = (
        book1.flatMap(lambda line: line.split())
        .map(lambda word: clean_word(word))
        .filter(lambda word: word and word not in stop_words)
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda x: x[1], ascending=False)
    )

    print("\nTop 25 words from book1.txt (extended):")
    print_top_n(book1_extended, 25)

    # Stop Spark
    sc.stop()
