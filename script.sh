#!/bin/bash
# shellcheck disable=SC1128
# Specifica il percorso della cartella
properties_folders=(
  "properties/properties_cartpole"
  "properties/properties_lunar"
  "properties/properties_dubins"
  "properties/properties_AC1"
  "properties/properties_AC2"
  "properties/properties_AC3"
  "properties/properties_AC4"
  "properties/properties_AC5"
  "properties/properties_AC6"
  "properties/properties_AC7"
  "properties/properties_AC8"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
  "properties/properties_ACAS"
)

networks=(
  "cartpole.onnx"
  "lunarlander.onnx"
  "dubinsrejoin.onnx"
  "AC1.onnx"
  "AC2.onnx"
  "AC3.onnx"
  "AC4.onnx"
  "AC5.onnx"
  "AC6.onnx"
  "AC7.onnx"
  "AC8.onnx"
  "ACAS_XU_1_1.onnx"
  "ACAS_XU_1_2.onnx"
  "ACAS_XU_2_1.onnx"
  "ACAS_XU_2_2.onnx"
  "ACAS_XU_3_1.onnx"
  "ACAS_XU_3_2.onnx"
  "ACAS_XU_4_1.onnx"
  "ACAS_XU_4_2.onnx"
  "ACAS_XU_5_1.onnx"
  "ACAS_XU_5_2.onnx"


)


for i in "${!networks[@]}"; do
  #echo "element $i is ${networks[$i]}"
    for file in "${properties_folders[$i]}"/*; do
        if [[ -f $file ]]; then
        # Stampa il nome del file
        python verify_network.py -n "${networks[$i]}" -p "$file"
        fi
    done
done
