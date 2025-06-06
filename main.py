from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Pre_Processing.data_preparation import extract_data
from Pre_Processing.dataset_wrapper import ImageWithAttentionDataset
from Model.model_utilities import ModelUtilities
import constants as constants

if __name__ == "__main__":
    # Extracting the data + applying pre-processing
    data = extract_data(apply_preprocessing=constants.APPLY_PREPROCESSING, scan_type=constants.CHOSEN_SCAN_TYPE)

    # Preparing the datasets
    train_df, test_df = train_test_split(data, test_size=constants.TEST_DATASET_SIZE, random_state=constants.RANDOM_STATE, shuffle=constants.SHUFFLE_DATA)
    train_dataset = ImageWithAttentionDataset(dataframe=train_df, scan_type=constants.CHOSEN_SCAN_TYPE)
    test_dataset = ImageWithAttentionDataset(dataframe=test_df, scan_type=constants.CHOSEN_SCAN_TYPE)

    train_loader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=False, num_workers=constants.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, shuffle=False, num_workers=constants.NUM_WORKERS)

    # Training the model
    model_utilities = ModelUtilities(constants.MODEL_TYPE)

    print(f"🧠 Training the model...")
    for epoch in range(1, constants.NUM_EPOCHS + 1):
        print(f"\n🎓 Epoch {epoch}/{constants.NUM_EPOCHS}")
        model_utilities.train_one_epoch(train_loader)

        print("\n📊 Final evaluation on test set:")
        model_utilities.evaluate_one_epoch(test_loader)