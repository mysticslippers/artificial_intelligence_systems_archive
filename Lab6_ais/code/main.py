def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def main(path: str = "diabetes.csv") -> None:
    try:
        data = load_data(path)
    except FileNotFoundError:
        print(f"Файл '{path}' не найден. Проверь путь к датасету.")
        return


if __name__ == "__main__":
    main()
