import torch

kernel_size = 2
padding = 0
stride = 1
filters_number = 1
batch_n = 3

kernel_param_size = kernel_size * kernel_size * 1

filter_weights = torch.arange(0, 4, dtype=torch.float32, device="cuda")
filter_weights = filter_weights * 10
inputs = torch.arange(0, 3 * 3 * 3).reshape(3, 1, 3, 3).to(torch.float32).to("cuda")
#filter_bias = params[1].squeeze()

input_shape = inputs.shape
batch_size = input_shape[0]

input_flattened = inputs.reshape(batch_size, -1)

image_shape = (input_shape[1], input_shape[2], input_shape[3])
image_flattened_dim = image_shape[1] * image_shape[2]

assert image_shape[1] == input_shape[2] and image_shape[0] == 1, "The image must be squared, 1 channel till now"

# Calcolare la dimensione dell'output dell'immagine a cui è stato applicato il filtro
output_conv_dim = ((image_shape[1] - kernel_size + 2 * padding) / stride) + 1
output_conv_shape = (output_conv_dim, output_conv_dim)
output_conv_shape_flatten = int(output_conv_dim * output_conv_dim)

# Array contenente gli indici
matrix_index = torch.arange(0, image_flattened_dim, dtype=torch.float32, device=de).reshape(1, image_shape[1], image_shape[1])

assert padding == 0, "Padding diverso da 0 non è supportato!"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Unfold per generare la matrice delle patch
patch_matrix = torch.nn.functional.unfold(matrix_index, kernel_size=kernel_size, stride=stride)
patch_matrix = patch_matrix.transpose(0, 1).to(torch.int32)

patch_number = patch_matrix.shape[0]  # Numero delle patch

# Crea la matrice di zeri (aggiungiamo la dimensione per i filtri)
zero_matrix = torch.zeros(patch_number, image_flattened_dim, filters_number, device=device)

# Ciclo sui filtri
for f_idx in range(filters_number):
    filter = filter_weights
    for i in range(patch_number):
        temp = torch.zeros(image_flattened_dim, dtype=torch.float32, device=device)
        indices = patch_matrix[i, :]  #
        temp[indices] = filter
        zero_matrix[i, :, f_idx] = temp

reshaped_matrix = zero_matrix.permute(0, 2, 1).reshape(-1, 9)

filter_weights = filter_weights.to(device)
output_tensor = torch.matmul(input_flattened, reshaped_matrix.T)  # (128, 2704)
pass