i = 0
should_exit = False

while i < 6:
    print(f"Before increment: {i}")
    if i == 5:
        print("Marking for exit!")
        should_exit = True
        i += 1
        continue
    i += 1
    print(f"After increment: {i}")

    if should_exit:
        print("Exiting the loop!")
        break

print("Loop finished.")