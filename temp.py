import pyautogui
import time
import math
import pandas as pd
import numpy as np
try:
    name = input("Enter your username (Please keep this consistent if you run this more than once as this will be used to create a label): ")
    numLoops= input("How many entries would you like to add (The more you do will make the model better): ")
    df = pd.DataFrame(
        columns=['dx1', 'dy1', 'dx2', 'dy2', 'dx3', 'dy3', 'dx4', 'dy4', 'dx5', 'dy5', 'dx6', 'dy6', 'dx7', 'dy7',
                 'dx8', 'dy8', 'dx9', 'dy9', 'dx10', 'dy10', 'Distance', 'Username'])
    for j in range(int(numLoops)):
        prevx,prevy=pyautogui.position()
        print(f"Start position: ({prevx},{prevy})")
        totalDistanceTravelled=0
        arr = []

        #give the user the chance to actually move the mouse
        time.sleep(1)
        for i in range(10):
            x, y = pyautogui.position()
            changex,changey=x-prevx,y-prevy
            print(f"Mouse position {i}: ({changex}, {changey})")
            totalDistanceTravelled+=math.sqrt(pow(changex,2)+pow(changey,2))
            prevx, prevy = x, y
            entry=[changex,changey]
            arr.append(entry)
            time.sleep(.3)
        print(f"Final array: {arr[0][0]}")
        print(f"Total Distance: {totalDistanceTravelled}")
        # I am hard coding this. Quite frankly the easiest way to do it
        data = {'dx1': arr[0][0], 'dy1': arr[0][1], 'dx2': arr[1][0], 'dy2': arr[1][1],
                'dx3': arr[2][0], 'dy3': arr[2][1], 'dx4': arr[3][0], 'dy4': arr[3][1],
                'dx5': arr[4][0], 'dy5': arr[4][1], 'dx6': arr[5][0], 'dy6': arr[5][1],
                'dx7': arr[6][0], 'dy7': arr[6][1], 'dx8': arr[7][0], 'dy8': arr[7][1],
                'dx9': arr[8][0], 'dy9': arr[8][1], 'dx10': arr[9][0], 'dy10': arr[9][1],
                'Distance': totalDistanceTravelled, 'Username': name}
        temp_df = pd.DataFrame([data])
        #Pandas sucks so much
        df = df._append(temp_df)
    df.to_csv('out.csv', index=False)
except KeyboardInterrupt:
    print("\nStopped.")