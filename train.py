def train_epoch(model, train_loader, criterion, optimizer):
  print('Training...')
  model.train()

  running_loss = 0.0
  total_predictions = 0.0
  correct_predictions = 0.0

  start_time = time.time()
  for batch_idx, (data, target) in enumerate(train_loader):

    optimizer.zero_grad() #.backward() accumulates gradients
    data = data.to(device)
    target = target.to(device) # all data & model on same device

    outputs = model(data)
    loss = criterion(outputs, target)
    running_loss += loss.item()

    loss.backward()
    optimizer.step()

    for i in range(len(outputs)):
      if torch.argmax(outputs[i]) == target[i]:
        correct_predictions += 1.0
      
      total_predictions += 1.0

  end_time = time.time()

  running_loss /= len(train_loader)
  print('Training Loss: ', running_loss, 'Time: ', end_time-start_time, 's')
  acc = (correct_predictions/total_predictions)*100.0
  print('Training Accuracy: ', acc, '%')
  return running_loss, acc


def val_model(model, val_loader, criterion):
  print('Validating...')
  with torch.no_grad():
    model.eval()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0

    for batch_idx, (data, target) in enumerate(val_loader):
      data = data.to(device)
      target = target.reshape(-1)
      target = target.to(device)
      outputs = model(data)
    
      for i in range(len(outputs)):
        if torch.argmax(outputs[i]) == target[i]:
          correct_predictions += 1.0
      
        total_predictions += 1.0

      loss = criterion(outputs, target).detach()
      running_loss += loss.item()
    
    running_loss /= len(val_loader)
    print('Testing Loss: ', running_loss)
    acc = (correct_predictions/total_predictions)*100.0
    print('Testing Accuracy: ', acc, '%')
    return running_loss, acc