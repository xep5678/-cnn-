

NE = 0.05
model_path = ''
for epoch in range(0, 100):
    model_path = ('epoch_{:.2f}'
                  .format((epoch + 1) * NE))

    if ('.50' not in model_path) and ('.00' not in model_path):
        print('del', model_path)


# for epoch in range(0, 100):
#     print(epoch)

str = '0.5epoch'
print('0.5' not in str)