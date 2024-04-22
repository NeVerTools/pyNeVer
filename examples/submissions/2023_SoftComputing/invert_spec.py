import os

property_path = '../../benchmarks/Drones/Properties'

lines = []

for property_file in os.listdir(property_path):
    with open(os.path.join(property_path, property_file), 'r') as fr:
        lines = fr.readlines()

    with open(os.path.join(property_path, property_file), 'w') as fw:
        ct = 0
        while True:
            fw.write(lines[ct])

            if lines[ct] == ';; --- OUTPUT CONSTRAINTS ---\n':
                ct += 1
                break

            ct += 1

        # First comes the or statement
        fw.write('(assert (or\n')

        # First line is (assert (<= (* 1.0 Y_0) VALUE))
        # Change '<=' with '>='
        fw.write('\t')
        fw.write(lines[ct].replace('<= (* 1.0 Y_0)', '>= Y_0').replace('(assert', '')[:-2])
        fw.write('\n')
        ct += 1

        # Second line is (assert (<= (* -1.0 Y_0) VALUE))
        # Change '(* -1.0 Y_0)' with 'Y_0' and invert VALUE
        last = lines[ct]
        last = last.replace('(* -1.0 Y_0)', 'Y_0').replace('(assert', '')[:-2]
        tokens = last.split(' ')
        value = tokens[-1]
        if value[0] == '-':
            value = value.replace('-', '')
        else:
            value = '-' + value

        tokens[-1] = value

        text = ''
        for t in tokens:
            text += f"{t} "
        fw.write(f"\t{text}\n")
        fw.write('))')
