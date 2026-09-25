
from src.evaluation.evaluator import DocumentAIEvaluator


# Actual results from the successful 20-page pipeline run
total_time = 718.27
total_pages = 20


evaluator = DocumentAIEvaluator()

metrics = evaluator.evaluate_performance(
    total_time=total_time,
    total_pages=total_pages
)


print("=" * 60)
print("DOCUMENT AI PERFORMANCE EVALUATION")
print("=" * 60)

print(
    f"Total Processing Time : "
    f"{metrics['total_processing_time_seconds']} seconds"
)

print(
    f"Total Pages           : "
    f"{metrics['total_pages']}"
)

print(
    f"Average Page Time     : "
    f"{metrics['average_page_time_seconds']} seconds"
)

print(
    f"Pages Per Second      : "
    f"{metrics['pages_per_second']}"
)

print("=" * 60)
print("PERFORMANCE EVALUATION COMPLETE")
print("=" * 60)
