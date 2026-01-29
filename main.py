from data import dataloader


def main():
    print("Loading Datasets")
    train, test = dataloader.get_datasets()
    print("Done")

if __name__ == "__main__":
    main()