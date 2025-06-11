import time
import pyautogui

# 持续的小时数：
# 11h = 39600 s
# 39600 / 600 = 66

for i in range(660):
  print(f'开始第{i}次点击.')
  # 打印鼠标当前位置
  # print(pyautogui.position())

  time.sleep(2)
  pyautogui.click(289, 723)
  time.sleep(2)
  pyautogui.click(1055, 720)

  # pyautogui.moveTo(x, y, 0.5)
  # pyautogui.moveTo(
  # 820, 690, 1)
  # print(f'i:{i}, x:{x}, y:{y}')zz
  # pyautogui.dragRel(0, -160, 0.5)

  # 间隔 10 分钟活动鼠标， 10 * 60 = 600
  time.sleep(60)