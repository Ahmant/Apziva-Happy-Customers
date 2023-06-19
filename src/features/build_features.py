from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def prepare_for_modeling(data, save_path = None):
    """
    Preprocesses the input data for modeling by performing the following steps:
    1. Renames the columns of the input data to standard names.
    2. Drops unimportant columns.
    3. Handles missing values (TODO: Implement this step).
    4. Splits the dataset into features (X) and target (y).
    5. Splits the dataset into train and test sets.
    6. Scales the feature values to a uniform range.

    Args:
        data (pandas.DataFrame): The input data containing features and target.
        save_path (string): The path where to save the processed/cleaned dataset.

    Returns:
        tuple: A tuple containing four elements:
            - X_train_rescaled (numpy.ndarray): Rescaled training features.
            - X_test_rescaled (numpy.ndarray): Rescaled testing features.
            - y_train (pandas.Series): Training target.
            - y_test (pandas.Series): Testing target.
    """

    data = data.copy()
    data.columns = ['is_happy', 'delivered_on_time', 'order_as_expected', 'order_everything', 'good_price', 'satisfied_with_courier', 'easy_with_app']

    # Drop unimportant columns
    unimportant_columns = ['order_as_expected']
    data.drop(columns=unimportant_columns, inplace=True)

    # TODO: Check nan values and clean/fix them

    # Save processed/cleaned dataset
    if save_path is not None:
        data.to_csv(save_path + '/data_cleaned.csv')

    # Splitting the dataset into "features" and "target"
    target_column = 'is_happy'
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the feature values to a uniform range
    scaler = MinMaxScaler()
    X_train_rescaled = scaler.fit_transform(X_train)
    X_test_rescaled = scaler.transform(X_test)

    return X_train_rescaled, X_test_rescaled, y_train, y_test
