def c_to_f(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def f_to_c(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9


def main() -> None:
    while True:
        print("=== Temperature Converter ===")
        print("1) Celsius to Fahrenheit")
        print("2) Fahrenheit to Celsius")
        print("Q) Quit")

        choice = input("Choose 1, 2 or Q: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break

        if choice == "1":
            try:
                value = float(input("Enter temperature in °C: ").strip())
                result = c_to_f(value)
                print(f"{value:.2f} °C = {result:.2f} °F\n")
            except ValueError:
                print("Please enter a valid number (e.g. 37 or 37.5).\n")

        elif choice == "2":
            try:
                value = float(input("Enter temperature in °F: ").strip())
                result = f_to_c(value)
                print(f"{value:.2f} °F = {result:.2f} °C\n")
            except ValueError:
                print("Please enter a valid number (e.g. 98.6).\n")

        else:
            print("Invalid choice, please type 1, 2 or Q.\n")


if __name__ == "__main__":
    main()