"""CLI entry: ``python3 -m data_validator`` runs the full validation and prints a summary."""

from . import DataValidator, KaggleDataIntegrator


def main() -> None:
    validator = DataValidator()
    report = validator.run_full_validation()

    report.print_summary()
    report.save_report("/tmp/data_validation_report.json")

    print("\n🔍 Discovering Kaggle datasets...")
    for ds in KaggleDataIntegrator.discover_relevant_datasets():
        print(f"  • {ds['title']}")
        print(f"    Coverage: {ds['coverage']}, Records: {ds['records']:,}")


if __name__ == "__main__":
    main()
