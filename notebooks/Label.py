# Example of CoNLL format labeling for a single sentence
labeled_data = """
ስልኩ  B-Product
የተለዋዋጭ  I-Product
ሳንባ  I-Product
ዋጋ  B-PRICE
1000  I-PRICE
ብር  I-PRICE
"""

# Save the labeled data to a file
with open("labeled_data.conll", "w") as file:
    file.write(labeled_data)
