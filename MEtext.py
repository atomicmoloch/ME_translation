import csv


filename = input("Enter input filename: ")


try:
    # Try to open the file in read mode
    with open(filename, 'r') as f:
        with open(filename + '.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile)

            while True:
                nln = f.readline()
                if not nln:
                    break
                split_nln = nln.split()
                if len(split_nln) > 0:
                    split_nln = nln.split()
                    print(split_nln)
                    wlen = int(input("Column 1? "))
                    if wlen < 0:
                        print("Entering manual mode")
                        while True:
                            c1 = input("Column 1 data? ")
                            if c1 == "":
                                break
                            c1 = c1.replace("P", "þ")
                            c1 = c1.replace("3", "ȝ")
                            c2 = input("Column 2 data? ")
                            c3 = input("Column 3 data? ")
                            writer.writerow([c1, c2, c3])
                    if wlen > 0:
                        glen = int(input("Column 2? ")) + wlen
                        dlen = int(input("Column 3? ")) + glen
                        word = " ".join(split_nln[0:wlen])
                        word = word.replace("P", "þ")
                        word = word.replace("3", "ȝ")
                        print(word)
                        gram = " ".join(split_nln[wlen:glen])
                        gram = gram.replace(".", "")
                        gram = gram.replace("7", "n")
                        print(gram)
                        defin = " ".join(split_nln[glen:dlen])
                        print(defin)

                        writer.writerow([word, gram, defin])



        f.close()
        outfile.close()

except FileNotFoundError:
    print(f"Sorry, the file {filename} does not exist.")
except IOError:
    print(f"Error opening file {filename}.")
