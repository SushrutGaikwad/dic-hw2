from pyspark import SparkContext, RDD
import string
import shutil
from pathlib import Path
from typing import List, Set


class WordCount:
    """
    A class to perform basic and extended word count using PySpark.
    """

    def __init__(self, input_paths: List[str], stop_words: Set[str] = set()) -> None:
        """
        Initializes the Spark context and loads the input files.

        Args:
            input_paths (List[str]): List of paths to the input text files.
            stop_words (Set[str], optional): A custom set of stop words to exclude
                in the extended word count. Defaults to an empty set, i.e., set().
        """
        self.sc = SparkContext("local", "WordCountOOP")
        self.input_paths = input_paths
        self.stop_words = stop_words
        self.text_rdd = self.load_data()

    def load_data(self) -> RDD:
        """
        Loads text files as a union of RDDs.

        Returns:
            RDD: RDD containing all lines from the input files.
        """
        rdds = [self.sc.textFile(path) for path in self.input_paths]
        return self.sc.union(rdds)

    def basic_word_count(self) -> RDD:
        """
        Performs a basic word count without cleaning or filtering.

        Returns:
            RDD: RDD of (word, count) pairs.
        """
        return (
            self.text_rdd.flatMap(lambda line: line.split())
            .map(lambda word: (word, 1))
            .reduceByKey(lambda a, b: a + b)
        )

    def clean_word(self, word: str) -> str:
        """
        Removes punctuation and converts a given word to lowercase.

        Args:
            word (str): Original word.

        Returns:
            str: Cleaned word.
        """
        return word.translate(str.maketrans("", "", string.punctuation)).lower()

    def extended_word_count(self, sort_desc: bool = True) -> RDD:
        """
        Performs extended word count: lowercase, punctuation removal, stop word
        removal, and sorting.

        Args:
            sort_desc (bool, optional): Sort descending by count. Defaults to True.

        Returns:
            RDD: RDD of (word, count) pairs.
        """
        return (
            self.text_rdd.flatMap(lambda line: line.split())
            .map(lambda word: self.clean_word(word))
            .filter(lambda word: word and word not in self.stop_words)
            .map(lambda word: (word, 1))
            .reduceByKey(lambda a, b: a + b)
            .sortBy(lambda x: x[1], ascending=not sort_desc)
        )

    def save_output(self, rdd: RDD, output_path: str) -> None:
        """
        Saves an RDD to a text file, removing the directory if it already exists.

        Args:
            rdd (RDD): The RDD to save.
            output_path (str): Path to the output folder.
        """
        path = Path(output_path)
        if path.exists() and path.is_dir():
            print(f"[INFO] Output path '{output_path}' exists. Deleting it first.")
            shutil.rmtree(path)

        rdd.saveAsTextFile(output_path)
        print(f"[INFO] Saved RDD to '{output_path}'")

    def print_top_n(self, rdd: RDD, n: int = 25) -> None:
        """
        Prints the top `n` words and their counts.

        Args:
            rdd (RDD): The RDD to take from.
            n (int, optional): Number of top results to print. Defaults to 25.
        """
        top_n = rdd.take(n)
        for word, count in top_n:
            print(f"{word}: {count}")

    def stop(self) -> None:
        """
        Stops the Spark context.
        """
        self.sc.stop()


if __name__ == "__main__":
    # Define a custom stop word list
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

    # Instantiate WordCount with both books
    wc = WordCount(input_paths=["book1.txt", "book2.txt"], stop_words=stop_words)

    # Task 1.1 - Basic Word Count
    basic_rdd = wc.basic_word_count()
    wc.save_output(basic_rdd, "output_1.txt")

    # Task 1.2 - Extended Word Count
    extended_rdd = wc.extended_word_count()
    wc.save_output(extended_rdd, "output_1_extended.txt")

    # Task 1.4 - Top 25 from book1.txt only
    book1_wc = WordCount(input_paths=["book1.txt"], stop_words=stop_words)
    book1_extended = book1_wc.extended_word_count()
    print("\nTop 25 words from book1.txt (extended):")
    book1_wc.print_top_n(book1_extended, n=25)

    # Stop SparkContexts
    wc.stop()
    book1_wc.stop()
