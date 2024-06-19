# Metrics operations

def calculate_SNR(model, image):
  model = model.eval()
  repetition = 30
  output_vec = torch.zeros((repetition,BATCH_SIZE,OUTPUT_SIZE))
  epsilon = 1e-15
  #index = obtain_index(label)
  snr_vec = torch.zeros((BATCH_SIZE,OUTPUT_SIZE))
  with torch.no_grad():
    for k in range(repetition):
      probs = model(image)
      # print('probs: ', probs[0])
      #index = np.argmax(probs)
      #output=model(image)[0][index]
      #output_vec[i] = probs[index]
      # output_vec[k] = np.sqrt(np.sum(np.square(probs.detach().numpy())))
      output_vec[k,:,:] = probs

  std_devs = torch.std(output_vec, axis=0)
  std_devs[std_devs == 0] = epsilon
  snr_vec = (torch.mean(output_vec, axis=0) / std_devs)
  # print('snr_vec: ',snr_vec[0])
  snr_log = 10*torch.log10(torch.mean(snr_vec,axis=1))
  # print('snr_log: ', snr_log)

  return torch.mean(snr_log)

def calculate_gaussianity(model, image):
  model = model.eval()
  dev = next(model.parameters()).device
  repetition = 30
  output_vec = torch.zeros((repetition,BATCH_SIZE,OUTPUT_SIZE))
  with torch.no_grad():
    for k in range(repetition):
      probs = model(image)
      # print('probs: ', probs)
      #index = np.argmax(probs)
      #output=model(image)[0][index]
      #output_vec[i] = probs[index]
      # output_vec[k] = np.sqrt(np.sum(np.square(probs.detach().numpy())))
      output_vec[k,:,:] = probs

  p_vec = (scipy.stats.normaltest(cp.asnumpy(output_vec), axis=0))[1]
  print('p_vec: ',p_vec)
  ratio = (sum(p > 0.05 for p in p_vec.flatten()))/(len(p_vec)*OUTPUT_SIZE)
  print('ratio: ', ratio)
  return torch.tensor(ratio,device=dev)


def calculate_entropy (model, image):
  model = model.eval()
  dev = next(model.parameters()).device
  k=50
  class_prob = torch.zeros((OUTPUT_SIZE),device=dev)
  entropy=0
  with torch.no_grad():
    for i in range(k):
      predicted_output = model(image)[0]
      for m in range(OUTPUT_SIZE):
        class_prob[m] += predicted_output[m]
  class_prob = class_prob / k
  for prob in class_prob:
    if prob != 0:  # Avoid log(0)
        entropy -= prob * torch.log(prob)
  return entropy

