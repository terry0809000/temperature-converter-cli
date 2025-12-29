def c_to_f(celsius: float) -> float:
    return celsius * 9 / 5 + 32

def f_to_c(fahrenheit: float) -> float:
    return (fahrenheit - 32) * 5 / 9

def main():
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
            value = input("Enter temperature in °C: ").strip()
            if value.replace('.', '', 1).isdigit():
                value = float(value)
                print(f"{value:.2f} °C = {c_to_f(value):.2f} °F\n")
            else:
                print("Invalid number. Try something like 37 or 37.5.\n")

        elif choice == "2":
            value = input("Enter temperature in °F: ").strip()
            if value.replace('.', '', 1).isdigit():
                value = float(value)
                print(f"{value:.2f} °F = {f_to_c(value):.2f} °C\n")
            else:
                print("Invalid number. Try something like 98.6.\n")

        else:
            print("Invalid choice, try again.\n")

if __name__ == "__main__":
    main()
