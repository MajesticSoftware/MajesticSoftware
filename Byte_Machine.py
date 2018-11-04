print("Welcome to the Byte Machine")

amount_of_bytes = input("How many bytes do you have?")

print("You have :" + amount_of_bytes + " bytes")

print("You may convert your bytes to:")
print("Kilobytes - 1024 bytes, enter 'A' ")
print("Megbytes - 1048576 bytes, enter 'B' ")
print("Gigabytes - 1073741824 bytes, enter 'C' ")
print("Terabyte - 1099511627776 bytes enter 'D' ")
toint = int(amount_of_bytes)
choice = input("Which would you like to convert to?: ")

if choice is "A":
    kilo = toint / 1024
    kilo = float(kilo)
    kilo = str(kilo)
    print("You have: " + kilo + " Kilobytes")
elif choice is "B":
    mega = toint / 1048576
    mega = float(mega)
    mega = str(mega)
    print("You have: " + mega + " Megabytes")
elif choice is "C":
    giga = toint / 1073741824
    giga = float(giga)
    giga = str(giga)
    print("You have: " + giga + " Gigabytes")
elif choice is "D":
    tera = toint / 1099511627776
    tera = float(tera)
    tera = str(tera)
    tera = print("You have: " + tera + " Terabytes")
else:
    print("You entered an invalid option, please restart the program and pick A, B ,C, or D")

print("Ending Program: Byte Machine...")
