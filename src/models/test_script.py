device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = Network()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

checkpoint_path = "../model_checkpoints/checkpoint.pt"
model, optimizer, epochs = load_checkpoint(checkpoint_path, model, optimizer)

test_data = ImageDataset("../../volume/test_data")
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, num_workers=0)
img_gray, img_ab, img_inception = iter(test_dataloader).next()
img_gray, img_ab, img_inception = img_gray.to(device), img_ab.to(device), img_inception.to(device)

model.eval()
with torch.no_grad():
    output = model(img_gray, img_inception)

for idx in range(100):
  grayscale, predicted_image, ground_truth = convert_to_rgb(
      img_gray[idx].cpu(), 
      output[idx].cpu(), 
      img_ab[idx].cpu()
  )
  img = cv2.convertScaleAbs(predicted_image, alpha=(255.0))
  cv2.imwrite(f'../../results/mae_results_V/output_{idx}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))