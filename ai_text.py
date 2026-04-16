from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

text = input("Enter your sentence: ")
result = classifier(text)

print("\nResult:")
print(f"Sentiment: {result[0]['label']}")
print(f"Confidence: {round(result[0]['score'], 2)}")