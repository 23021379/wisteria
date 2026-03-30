import csv
from math import e
import re
import ast
import time
from scipy.special import j0
from difflib import SequenceMatcher


parsed_data=[]

with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        parsed_row = []
        for cell in row:
            try:
                # Convert the string representation of a list to an actual list
                parsed_cell = ast.literal_eval(cell.strip())
            except (SyntaxError, ValueError):
                # Fallback to raw string if parsing fails
                parsed_cell = cell.strip()
            parsed_row.append(parsed_cell)
        if parsed_row not in parsed_data:
            parsed_data.append(parsed_row)

parsed_data3=[]
with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        parsed_row = []
        for cell in row:
            try:
                # Convert the string representation of a list to an actual list
                parsed_cell = ast.literal_eval(cell.strip())
            except (SyntaxError, ValueError):
                # Fallback to raw string if parsing fails
                parsed_cell = cell.strip()
            parsed_row.append(parsed_cell)
        if parsed_row not in parsed_data:
            parsed_data3.append(parsed_row)

for i in range(1, (len(parsed_data)-1)):
    Lcheck=[]
    actualSold=""
    actualDate=""
    for j in range(len(parsed_data3)):
        matcher = SequenceMatcher(None, str(parsed_data3[j][0].lower().replace("-","")), str(parsed_data[i][1][0].lower().replace("-","")))
        if matcher.ratio()>0.7:
            actualSold=parsed_data3[j][1][1:].replace(",","")
            actualDate=parsed_data3[j][2]
            try:
                actualDate = actualDate[actualDate.find(" ")+1:]
            except:
                pass
            month_L=["january","february","march","april","may","june","july","august","september","october","november","december"]
            parsed_dataout_replace=actualDate[:actualDate.index(" ")]
            parsed_dataout_replace2=parsed_dataout_replace[:actualDate.index(" ")]
            for k in range(len(month_L)):
                if parsed_dataout_replace.lower() in month_L[k].lower():
                    actualDate=str(k+1)+"-"+actualDate[(actualDate.index(" ")+1):]
                    break
                elif parsed_dataout_replace2.lower() in month_L[k].lower():
                    actualDate=str(k+1)+"-"+actualDate[(actualDate.index(" ")+1):]
                    break
            break
            
    parsed_dataout=[]
    parsed_dataout=[str(parsed_data[i][0][0]), str(parsed_data[i][1][0]), str(actualSold), str(actualDate), str(parsed_data[i][1][1]), str(parsed_data[i][1][2]), str(parsed_data[i][1][3]), str(parsed_data[i][1][4]), str(parsed_data[i][1][5]), str(parsed_data[i][1][6]), str(parsed_data[i][1][7])]
    if "Sold" in parsed_data[i][2][0]:
        parsed_dataout.append(str("2"))
    elif "Listed" in parsed_data[i][2][0]:
        parsed_dataout.append(str("1"))
    else:
        parsed_dataout.append(str("0"))
    for j in range(len(parsed_data[i][2])):
        parsed_dataout.append(str(parsed_data[i][2][j][1]))
        parsed_dataout.append(str(parsed_data[i][2][j][2].replace("£","").replace(",","")))
        parsed_dataout.append(str(parsed_data[i][2][j][3].replace(" bed","").replace("s","")))
        parsed_dataout.append(str(parsed_data[i][2][j][4].replace(" bath","").replace("s","")))
        parsed_dataout.append(str(parsed_data[i][2][j][5].replace(" reception","").replace("s","")))
    if len(parsed_data[i][2]) < 15:
        for j in range((15-len(parsed_data[i][2]))):
            parsed_dataout.append(str(""))
            parsed_dataout.append(str(""))
            parsed_dataout.append(str(""))
            parsed_dataout.append(str(""))
            parsed_dataout.append(str(""))
    try:
        parsed_dataout.append(str(parsed_data[i][3][0]))
    except:
        parsed_dataout.append(str(""))
    try:
        parsed_dataout.append(str(parsed_data[i][3][1]))
    except:
        parsed_dataout.append(str(""))

    try:
        total_months = 0
        matches = re.findall(r'[REDACTED_BY_SCRIPT]', parsed_data[i][3][2], flags=re.IGNORECASE)
        for value_str, unit in matches:
            value = int(value_str)
            if unit.lower().startswith('year'):
                total_months += value * 12
            elif unit.lower().startswith('month'):
                total_months += value
        parsed_dataout.append(str(total_months))
    except:
        parsed_dataout.append(str(""))

    try:
        parsed_dataout.append(str(parsed_data[i][3][3].replace("£","").replace(",","")))
    except:
        parsed_dataout.append(str(""))

    try:
        if len(parsed_data[i][4])==0:
            parsed_dataout.append(str(""))
            parsed_dataout.append(str(""))
        elif len(parsed_data[i][4])==2:
            total_months = 0
            matches = re.findall(r'[REDACTED_BY_SCRIPT]', parsed_data[i][4][0], flags=re.IGNORECASE)
            for value_str, unit in matches:
                value = int(value_str)
                if unit.lower().startswith('year'):
                    total_months += value * 12
                elif unit.lower().startswith('month'):
                    total_months += value
            parsed_dataout.append(str(total_months))
            parsed_dataout.append(str(parsed_data[i][4][1]))
        else:
            parsed_dataout.append(str(""))
            parsed_dataout.append(str(parsed_data[i][4][0]))
    except:
        parsed_dataout.append(str(""))
        parsed_dataout.append(str(""))

    ##print("1",len(parsed_dataout))
    parsed_data2=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    # Convert the string representation of a list to an actual list
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    # Fallback to raw string if parsing fails
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data2:
                parsed_data2.append(parsed_row)
    found=False
    for j in range((len(parsed_data2))):
        try:
            if parsed_data2[j][0][0].lower().replace(" ","").replace(",","").replace("-","")==parsed_data[i][0][0].lower().replace(" ","").replace(",","").replace("-",""):
                parsed_dataout.append(parsed_data2[j][0][0])
                parsed_dataout.append(parsed_data2[j][1][0])
                alter= parsed_data2[j][3][1].replace("£","").replace(",","").replace(" ","")
                alter= alter.split("-")
                parsed_dataout.append(alter[0])
                parsed_dataout.append(alter[1])
                parsed_dataout.append(parsed_data2[j][4][0])
                parsed_dataout.append(parsed_data2[j][5][0])
                parsed_dataout.append(parsed_data2[j][6][0])
                alter=parsed_data2[j][7][1]
                if "house" in alter.lower():
                    parsed_dataout.append(str("3"))
                elif "bungalow" in alter.lower():
                    parsed_dataout.append(str("2"))
                elif "flat" in alter.lower():
                    parsed_dataout.append(str("1"))
                else:
                    parsed_dataout.append(str("0"))
                for k in range(8,16):
                    parsed_dataout.append(parsed_data2[j][k][0])
                found=True
                break
        except:pass
    if found == False:
        for j in range(16):
            parsed_dataout.append("")

    #print(found, "2",len(parsed_dataout))
    #Lcheck.append(["1",len(parsed_dataout)])
    parsed_data2=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    # Convert the string representation of a list to an actual list
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    # Fallback to raw string if parsing fails
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data2:
                parsed_data2.append(parsed_row)

    found=False
    for j in range((len(parsed_data2))):
        if parsed_data2[j][0][0].lower().replace(" ","").replace(",","").replace("-","")==parsed_data[i][0][0].lower().replace(" ","").replace(",","").replace("-",""):
            try:
                parsed_dataout.append(parsed_data2[j][0][2])
            except:
                parsed_dataout.append("")
            for k in range(4,21):
                try:
                    parsed_dataout.append(parsed_data2[j][0][k])
                except:
                    parsed_dataout.append("")
            try:
                parsed_dataout.append([k.replace("%","") for k in parsed_data2[j][0][22]])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append([k.replace("%","") for k in parsed_data2[j][0][23]])
            except:
                parsed_dataout.append("")
            for k in range(24,29):
                try:
                    parsed_dataout.append(parsed_data2[j][0][k])
                except:
                    parsed_dataout.append("")
            found=True
            break
    if found == False:
        for j in range(25):
            parsed_dataout.append("")
    parsed_dataout[127]=['','','','','','','','','','','','','','','']
    parsed_dataout[128]=['','','','','','','','','','','','','','','']


    #print(found, "3",len(parsed_dataout))
    #Lcheck.append(["2",len(parsed_dataout)])
    parsed_data2=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    # Convert the string representation of a list to an actual list
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    # Fallback to raw string if parsing fails
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data2:
                parsed_data2.append(parsed_row)

    found=False
    #Lcheck.append(["3",len(parsed_dataout)])
    for j in range((len(parsed_data2))):
        if parsed_data2[j][0][0].lower().replace(" ","").replace(",","").replace("-","")==parsed_data[i][0][0].lower().replace(" ","").replace(",","").replace("-",""):
            try:
                parsed_dataout.append(parsed_data2[j][0][1][0])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][1][1])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][2])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][3])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][4])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][5])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][6])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][7])
            except:
                parsed_dataout.append("")
            try:
                parsed_dataout.append(parsed_data2[j][0][10][0])
                parsed_dataout.append(parsed_data2[j][0][10][1])
            except:
                parsed_dataout.append("")
                parsed_dataout.append("")
            #Lcheck.append(["4",len(parsed_dataout)])

            try:
                if "sqft" in parsed_data2[j][0][12][0].lower() and "sqft" in parsed_data2[j][0][11][0].lower():
                    parsed_dataout.append(parsed_data2[j][0][12][0].replace("sqft","").replace(",",""))
                    parsed_dataout.append(parsed_data2[j][0][12][1].replace("sqft","").replace(",",""))
                elif "sqft" in parsed_data2[j][0][11].lower():
                    parsed_dataout.append(parsed_data2[j][0][11].replace("sqft","").replace(",",""))
                    parsed_dataout.append("")
                else:
                    parsed_dataout.append("")
                    parsed_dataout.append("")
            except:
                parsed_dataout.append("")
                parsed_dataout.append("")
            #Lcheck.append(["5",len(parsed_dataout)])
            for k in range(13,21):
                try:
                    parsed_dataout.append(parsed_data2[j][0][k])
                except:
                    parsed_dataout.append("")
            #Lcheck.append(["6",len(parsed_dataout)])
            try:
                if "high" in parsed_data2[j][0][22].lower():
                    parsed_dataout.append(str("3"))
                elif "very-low" in parsed_data2[j][0][22].lower():
                    parsed_dataout.append(str("2"))
                elif "low" in parsed_data2[j][0][22].lower() and "very-low" not in parsed_data2[j][0][22].lower():
                    parsed_dataout.append(str("1"))
                else:
                    parsed_dataout.append(str("0"))
            except:
                parsed_dataout.append(str(""))
            #Lcheck.append(["7",len(parsed_dataout)])
            for k in range(24,32):
                try:
                    parsed_dataout.append(parsed_data2[j][0][k])
                except:
                    parsed_dataout.append("")
            #Lcheck.append(["8",len(parsed_dataout)])
            try:
                if "high" in parsed_data2[j][0][33].lower():
                    parsed_dataout.append(str("3"))
                elif "very-low" in parsed_data2[j][0][33].lower():
                    parsed_dataout.append(str("2"))
                elif "low" in parsed_data2[j][0][33].lower() and "very-low" not in parsed_data2[j][0][33].lower():
                    parsed_dataout.append(str("1"))
                else:
                    parsed_dataout.append(str("0"))
            except:
                parsed_dataout.append(str(""))
            #Lcheck.append(["9",len(parsed_dataout)])
            for k in range(35,43):
                try:
                    parsed_dataout.append(parsed_data2[j][0][k])
                except:
                    parsed_dataout.append("")
            #Lcheck.append(["10",len(parsed_dataout)])
            try:
                if "high" in parsed_data2[j][0][44].lower():
                    parsed_dataout.append(str("3"))
                elif "very-low" in parsed_data2[j][0][44].lower():
                    parsed_dataout.append(str("2"))
                elif "low" in parsed_data2[j][0][44].lower() and "very-low" not in parsed_data2[j][0][44].lower():
                    parsed_dataout.append(str("1"))
                else:
                    parsed_dataout.append(str("0"))
            except:
                parsed_dataout.append(str(""))
            found = True
    #Lcheck.append(["11",len(parsed_dataout)])
    if found == False:
        for j in range((169-len(parsed_dataout))):
            parsed_dataout.append("")
        if len(parsed_dataout) > 169:
            for k in range(len(parsed_dataout)-169,0,1):
                parsed_dataout.pop(141+k)
    else:
        if len(parsed_dataout) > 169:
            for k in range(len(parsed_dataout)-169):
                parsed_dataout.pop(141+k)


    #print(found, "4",len(parsed_dataout))
    #Lcheck.append(["12",len(parsed_dataout)])
    parsed_data2=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    # Convert the string representation of a list to an actual list
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    # Fallback to raw string if parsing fails
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data2:
                parsed_data2.append(parsed_row)

    found=False
    for j in range((len(parsed_data2))):
        if parsed_data2[j][0][0].lower().replace(" ","").replace(",","").replace("-","")==parsed_data[i][0][0].lower().replace(" ","").replace(",","").replace("-",""):
            try:
                if "high" in parsed_data2[j][0][1].lower():
                    parsed_dataout.append(str("5"))
                elif "medium-high" in parsed_data2[j][0][1].lower():
                    parsed_dataout.append(str("4"))
                elif "medium" in parsed_data2[j][0][1].lower():
                    parsed_dataout.append(str("3"))
                elif "low-medium" in parsed_data2[j][0][1].lower():
                    parsed_dataout.append(str("2"))
                elif "lower" in parsed_data2[j][0][1].lower():
                    parsed_dataout.append(str("1"))
                else:
                    parsed_dataout.append(str("0"))
            except:
                parsed_dataout.append("")
            #Lcheck.append(["12.1",len(parsed_dataout)])
            try:
                parsed_dataout.append(parsed_data2[j][0][2])
                parsed_dataout.append(parsed_data2[j][0][3])
            except:pass
            try:
                parsed_dataout.append(int(parsed_data2[j][0][4][0]))
            except:
                
                try:
                    if "A" in parsed_data2[j][0][4][0]:
                        parsed_dataout.append(str("6"))
                    elif "B" in parsed_data2[j][0][4][0]:
                        parsed_dataout.append(str("5"))
                    elif "C" in parsed_data2[j][0][4][0]:
                        parsed_dataout.append(str("4"))
                    elif "D" in parsed_data2[j][0][4][0]:
                        parsed_dataout.append(str("3"))
                    elif "E" in parsed_data2[j][0][4][0]:
                        parsed_dataout.append(str("2"))
                    elif "F" in parsed_data2[j][0][4][0]:
                        parsed_dataout.append(str("1"))
                    else:
                        parsed_dataout.append(str("0"))
                except:
                    parsed_dataout.append("")
            #Lcheck.append(["12.2",len(parsed_dataout)])
            try:
                if "semi-detached" in parsed_data2[j][0][4][2].lower():
                    parsed_dataout.append(str("3"))
                elif "detached" in parsed_data2[j][0][4][2].lower():
                    parsed_dataout.append(str("4"))
                elif "terraced" in parsed_data2[j][0][4][2].lower():
                    parsed_dataout.append(str("2"))
                elif "flat" in parsed_data2[j][0][4][2].lower():
                    parsed_dataout.append(str("1"))
                else:
                    try:
                        parsed_dataout.append(parsed_data2[j][0][4][2])
                    except:
                        parsed_dataout.append(str("0"))
            except:
                parsed_dataout.append("")
            #Lcheck.append(["12.3",len(parsed_dataout)])
            try:
                parsed_dataout.append(int(parsed_data2[j][0][4][3]))
            except:
                try:
                    if "semi-detached" in parsed_data2[j][0][4][2].lower():
                        parsed_dataout.append(str("3"))
                    elif "detached" in parsed_data2[j][0][4][2].lower():
                        parsed_dataout.append(str("4"))
                    elif "terraced" in parsed_data2[j][0][4][2].lower():
                        parsed_dataout.append(str("2"))
                    elif "flat" in parsed_data2[j][0][4][2].lower():
                        parsed_dataout.append(str("1"))
                    else:
                        try:
                            parsed_dataout.append(parsed_data2[j][0][4][2])
                        except:
                            parsed_dataout.append(str("0"))
                except:
                    parsed_dataout.append("")
            #Lcheck.append(["12.4",len(parsed_dataout)])
            try:
                parsed_dataout.append(int(parsed_data2[j][0][4][4][6:10]))
            except:
                parsed_dataout.append("")
            try:
                if "A" in parsed_data2[j][0][4][0]:
                    parsed_dataout.append(str("6"))
                elif "B" in parsed_data2[j][0][4][0]:
                    parsed_dataout.append(str("5"))
                elif "C" in parsed_data2[j][0][4][0]:   
                    parsed_dataout.append(str("4"))
                elif "D" in parsed_data2[j][0][4][0]:
                    parsed_dataout.append(str("3"))
                elif "E" in parsed_data2[j][0][4][0]:
                    parsed_dataout.append(str("2"))
                elif "F" in parsed_data2[j][0][4][0]:
                    parsed_dataout.append(str("1"))
                else:
                    parsed_dataout.append(str("0"))
            except:
                parsed_dataout.append("")
            #Lcheck.append(["12.5",len(parsed_dataout)])
            for k in range(6):
                try:
                    if "higher" in parsed_data2[j][0][5][k].lower():
                        parsed_dataout.append(str("3"))
                    elif "typical" in parsed_data2[j][0][5][k].lower():
                        parsed_dataout.append(str("2"))
                    elif "lower" in parsed_data2[j][0][5][k].lower():
                        parsed_dataout.append(str("1"))
                    else:
                        parsed_dataout.append(str("0"))
                except:
                    parsed_dataout.append(str(""))
            #Lcheck.append(["12.6",len(parsed_dataout)])
            found=True
            break
    if found==False:
        for j in range(183-len(parsed_dataout)):
            parsed_dataout.append("")
    #Lcheck.append(["13",len(parsed_dataout)])
    #print(found, "5",len(parsed_dataout))
    month_L=["january","february","march","april","may","june","july","august","september","october","november","december"]
    for j in range(10,90):
        try:
            parsed_dataout_replace=parsed_dataout[j][:parsed_dataout[j].index(" ")]
            if parsed_dataout_replace.lower() in month_L:
                parsed_dataout_replace = str(month_L.index(parsed_dataout_replace.lower())+1)
                parsed_dataout[j]=parsed_dataout_replace+"-"+parsed_dataout[j][(parsed_dataout[j].index(" ")+1):]
            else:
                parsed_dataout[j]=""
        except:
            parsed_dataout[j]=""
    #Lcheck.append(["14",len(parsed_dataout)])
    try:
        alter_temp=parsed_dataout[127]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("Owned")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Shared")]
                except:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Rented")]
        parsed_dataout[127]=alter_temp
        alter_temp=parsed_dataout[128]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("School")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Degree")]
                except:
                    try:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Other")]
                    except:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Student")]
        parsed_dataout[128]=alter_temp
    except:pass
    try:
        alter_temp=parsed_dataout[126]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("Owned")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Shared")]
                except:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Rented")]
        parsed_dataout[126]=alter_temp
        alter_temp=parsed_dataout[127]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("School")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Degree")]
                except:
                    try:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Other")]
                    except:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Student")]
        parsed_dataout[127]=alter_temp
    except:pass
    try:
        alter_temp=parsed_dataout[128]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("Owned")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Shared")]
                except:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Rented")]
        parsed_dataout[128]=alter_temp
        alter_temp=parsed_dataout[129]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("School")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Degree")]
                except:
                    try:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Other")]
                    except:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Student")]
        parsed_dataout[129]=alter_temp
    except:pass


    try:
        alter_temp=parsed_dataout[125]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("Owned")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Shared")]
                except: 
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Rented")]
        parsed_dataout[125]=alter_temp
        alter_temp=parsed_dataout[126]
        for j in range(len(alter_temp)):
            try:
                alter_temp[j]=alter_temp[j][:alter_temp[j].index("School")]
            except:
                try:
                    alter_temp[j]=alter_temp[j][:alter_temp[j].index("Degree")]
                except:
                    try:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Other")]
                    except:
                        alter_temp[j]=alter_temp[j][:alter_temp[j].index("Student")]
        parsed_dataout[126]=alter_temp
    except:pass
    Lcheck.append(["15",len(parsed_dataout)])
    parsed_data2=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    # Convert the string representation of a list to an actual list
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    # Fallback to raw string if parsing fails
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data2:
                parsed_data2.append(parsed_row)
    #utility room, kitchen, Dining Room, Kitchen/Diner, living/dining room, Living Room, primary bedroom, bedroom, Bedroom 1, Bedroom One, Bedroom 2, Bedroom Two, Bedroom 3, Bedroom 4, Bedroom 5, Bedroom 6, Bedroom 7, Bedroom 8, Lounge, Hall, hallway, closet, 
    #Landing, Reception Room, Kitchen/Utility, Master Bedroom, Shower Room, Ensuite, Robe, Sun Room, En-suite Bathroom, Entrance Hall, Inner Porch, Family Bathroom, En-suite Shower Room, Sitting Room/Play Room, Dining Area/Family Room, 
    #Kitchen/Breakfast Room, Open Plan Family Room-Diner, Garage, stairs/hallway, wardrobe, rear porch, lobby, Open Plan Lounge-Kitchen-Diner, Porch, Study, estimated other areas, Vestibule, Family Room,Jack and Jill En-suite, Sitting Room,
    #Cupboard, Rear Entrance, Conservatory, Staircase, Store, 
    #
    roomsList=["","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
    roomsList2=["","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
    roomsListCheck=["utility","kitchen","Dining", "diner","Living","Family", "Sitting", "Lounge", "Bedroom", "Hall", "Landing","closet", "Robe", "Reception", "Shower", "Bathroom", "Ensuite", "En-suite", "Sun Room", "Conservatory", "Porch", "Garage", "lobby", "Vestibule", "Study", "Cupboard", "Store"]
    for j in range(len(parsed_data2)):
        if parsed_data2[j][1].replace(",","").replace(" ","").replace("-","").lower()==parsed_data[i][0][0].replace(",","").replace(" ","").replace("-","").lower():
            for h in range(len(parsed_data2[j][3])):
                for k in range(len(roomsListCheck)):
                    if roomsListCheck[k].lower() in parsed_data2[j][3][h][0].lower():
                        if "/" in parsed_data2[j][3][h][0] or "-" in parsed_data2[j][3][h][0].lower():
                            if k == 1:
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[1]=parsed_data2[j][3][h][2]
                                        roomsList2[1]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[1]=parsed_data2[j][3][h][1]
                                        roomsList2[1]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[1]=parsed_data2[j][3][h][1]
                                    roomsList2[1]=parsed_data2[j][3][h][2]
                            elif k == 2 or k==3:
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[3]=parsed_data2[j][3][h][2]
                                        roomsList2[3]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[3]=parsed_data2[j][3][h][1]
                                        roomsList2[3]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[3]=parsed_data2[j][3][h][1]
                                    roomsList2[3]=parsed_data2[j][3][h][2]
                            elif k == 4 or k==5 or k==6:
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[5]=parsed_data2[j][3][h][2]
                                        roomsList2[5]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[5]=parsed_data2[j][3][h][1]
                                        roomsList2[5]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[5]=parsed_data2[j][3][h][1]
                                    roomsList2[5]=parsed_data2[j][3][h][2]
                        else:
                            if k == 1:
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[0]=parsed_data2[j][3][h][2]
                                        roomsList2[0]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[0]=parsed_data2[j][3][h][1]
                                        roomsList2[0]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[0]=parsed_data2[j][3][h][1]
                                    roomsList2[0]=parsed_data2[j][3][h][2]
                            elif k == 2 or k==3:
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[2]=parsed_data2[j][3][h][2]
                                        roomsList2[2]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[2]=parsed_data2[j][3][h][1]
                                        roomsList2[2]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[2]=parsed_data2[j][3][h][1]
                                    roomsList2[2]=parsed_data2[j][3][h][2]
                            elif k == 4 or k==5 or k==6 or k==7:
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[4]=parsed_data2[j][3][h][2]
                                        roomsList2[4]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[4]=parsed_data2[j][3][h][1]
                                        roomsList2[4]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[4]=parsed_data2[j][3][h][1]
                                    roomsList2[4]=parsed_data2[j][3][h][2]
                        if k==8:
                            if "1" in parsed_data2[j][3][h][0].lower() or "one" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[6]=parsed_data2[j][3][h][2]
                                        roomsList2[6]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[6]=parsed_data2[j][3][h][1]
                                        roomsList2[6]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[6]=parsed_data2[j][3][h][1]
                                    roomsList2[6]=parsed_data2[j][3][h][2]
                            elif "2" in parsed_data2[j][3][h][0].lower() or "two" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":  
                                        roomsList[7]=parsed_data2[j][3][h][2]
                                        roomsList2[7]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[7]=parsed_data2[j][3][h][1]
                                        roomsList2[7]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[7]=parsed_data2[j][3][h][1]
                                    roomsList2[7]=parsed_data2[j][3][h][2]
                            elif "3" in parsed_data2[j][3][h][0].lower() or "three" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[8]=parsed_data2[j][3][h][2]
                                        roomsList2[8]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[8]=parsed_data2[j][3][h][1]
                                        roomsList2[8]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[8]=parsed_data2[j][3][h][1]
                                    roomsList2[8]=parsed_data2[j][3][h][2]
                            elif "4" in parsed_data2[j][3][h][0].lower() or "four" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[9]=parsed_data2[j][3][h][2]
                                        roomsList2[9]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[9]=parsed_data2[j][3][h][1]
                                        roomsList2[9]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[9]=parsed_data2[j][3][h][1]
                                    roomsList2[9]=parsed_data2[j][3][h][2]
                            elif "5" in parsed_data2[j][3][h][0].lower() or "five" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":  
                                        roomsList[10]=parsed_data2[j][3][h][2]
                                        roomsList2[10]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[10]=parsed_data2[j][3][h][1]
                                        roomsList2[10]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[10]=parsed_data2[j][3][h][1]
                                    roomsList2[10]=parsed_data2[j][3][h][2]
                            elif "6" in parsed_data2[j][3][h][0].lower() or "six" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[11]=parsed_data2[j][3][h][2]
                                        roomsList2[11]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[11]=parsed_data2[j][3][h][1]
                                        roomsList2[11]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[11]=parsed_data2[j][3][h][1]
                                    roomsList2[11]=parsed_data2[j][3][h][2]
                            elif "7" in parsed_data2[j][3][h][0].lower() or "seven" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[12]=parsed_data2[j][3][h][2]
                                        roomsList2[12]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[12]=parsed_data2[j][3][h][1]
                                        roomsList2[12]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[12]=parsed_data2[j][3][h][1]
                                    roomsList2[12]=parsed_data2[j][3][h][2]
                            elif "8" in parsed_data2[j][3][h][0].lower() or "eight" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[13]=parsed_data2[j][3][h][2]
                                        roomsList2[13]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[13]=parsed_data2[j][3][h][1]
                                        roomsList2[13]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[13]=parsed_data2[j][3][h][1]
                                    roomsList2[13]=parsed_data2[j][3][h][2]
                            elif "9" in parsed_data2[j][3][h][0].lower() or "nine" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[14]=parsed_data2[j][3][h][2]
                                        roomsList2[14]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[14]=parsed_data2[j][3][h][1]
                                        roomsList2[14]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[14]=parsed_data2[j][3][h][1]
                                    roomsList2[14]=parsed_data2[j][3][h][2]
                            elif "10" in parsed_data2[j][3][h][0].lower() or "ten" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[15]=parsed_data2[j][3][h][2]
                                        roomsList2[15]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[15]=parsed_data2[j][3][h][1]
                                        roomsList2[15]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[15]=parsed_data2[j][3][h][1]
                                    roomsList2[15]=parsed_data2[j][3][h][2]
                            elif "main" in parsed_data2[j][3][h][0].lower() or "primary" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[16]=parsed_data2[j][3][h][2]
                                        roomsList2[16]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[16]=parsed_data2[j][3][h][1]
                                        roomsList2[16]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[16]=parsed_data2[j][3][h][1]
                                    roomsList2[16]=parsed_data2[j][3][h][2]
                        if k==9 or k==10:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[17]=parsed_data2[j][3][h][2]
                                    roomsList2[17]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[17]=parsed_data2[j][3][h][1]
                                    roomsList2[17]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[17]=parsed_data2[j][3][h][1]
                                roomsList2[17]=parsed_data2[j][3][h][2]
                        if k==11 or k==12:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[18]=parsed_data2[j][3][h][2]
                                    roomsList2[18]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[18]=parsed_data2[j][3][h][1]
                                    roomsList2[18]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[18]=parsed_data2[j][3][h][1]
                                roomsList2[18]=parsed_data2[j][3][h][2]
                        if k==13:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[19]=parsed_data2[j][3][h][2]
                                    roomsList2[19]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[19]=parsed_data2[j][3][h][1]
                                    roomsList2[19]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[19]=parsed_data2[j][3][h][1]
                                roomsList2[19]=parsed_data2[j][3][h][2]
                        if k==14 or k==15:
                            if "ensuite" in parsed_data2[j][3][h][0].lower():
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[20]=parsed_data2[j][3][h][2]
                                        roomsList2[20]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[20]=parsed_data2[j][3][h][1]
                                        roomsList2[20]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[20]=parsed_data2[j][3][h][1]
                                    roomsList2[20]=parsed_data2[j][3][h][2]
                            else:
                                try:
                                    if parsed_data2[j][3][h][1].lower() == "area":
                                        roomsList[21]=parsed_data2[j][3][h][2]
                                        roomsList2[21]=parsed_data2[j][3][h][3]
                                    else:
                                        roomsList[21]=parsed_data2[j][3][h][1]
                                        roomsList2[21]=parsed_data2[j][3][h][2]
                                except:
                                    roomsList[21]=parsed_data2[j][3][h][1]
                                    roomsList2[21]=parsed_data2[j][3][h][2]
                        if k==16 or k==17:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[20]=parsed_data2[j][3][h][2]
                                    roomsList2[20]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[20]=parsed_data2[j][3][h][1]
                                    roomsList2[20]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[20]=parsed_data2[j][3][h][1]
                                roomsList2[20]=parsed_data2[j][3][h][2]
                        if k ==18 or k==19:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[22]=parsed_data2[j][3][h][2]
                                    roomsList2[22]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[22]=parsed_data2[j][3][h][1]
                                    roomsList2[22]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[22]=parsed_data2[j][3][h][1]
                                roomsList2[22]=parsed_data2[j][3][h][2]
                        if k==20:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[23]=parsed_data2[j][3][h][2]
                                    roomsList2[23]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[23]=parsed_data2[j][3][h][1]
                                    roomsList2[23]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[23]=parsed_data2[j][3][h][1]
                                roomsList2[23]=parsed_data2[j][3][h][2]
                        if k==21:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[24]=parsed_data2[j][3][h][2]
                                    roomsList2[24]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[24]=parsed_data2[j][3][h][1]
                                    roomsList2[24]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[24]=parsed_data2[j][3][h][1]
                                roomsList2[24]=parsed_data2[j][3][h][2]
                        if k==22 or k==23:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[25]=parsed_data2[j][3][h][2]
                                    roomsList2[25]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[25]=parsed_data2[j][3][h][1]
                                    roomsList2[25]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[25]=parsed_data2[j][3][h][1]
                                roomsList2[25]=parsed_data2[j][3][h][2]
                        if k==24:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[26]=parsed_data2[j][3][h][2]
                                    roomsList2[26]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[26]=parsed_data2[j][3][h][1]
                                    roomsList2[26]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[26]=parsed_data2[j][3][h][1]
                                roomsList2[26]=parsed_data2[j][3][h][2]
                        if k==25 or k==26:
                            try:
                                if parsed_data2[j][3][h][1].lower() == "area":
                                    roomsList[27]=parsed_data2[j][3][h][2]
                                    roomsList2[27]=parsed_data2[j][3][h][3]
                                else:
                                    roomsList[27]=parsed_data2[j][3][h][1]
                                    roomsList2[27]=parsed_data2[j][3][h][2]
                            except:
                                roomsList[27]=parsed_data2[j][3][h][1]
                                roomsList2[27]=parsed_data2[j][3][h][2]
            break
    for j in range(len(roomsList)):
        parsed_dataout.append(roomsList[j])
        parsed_dataout.append(roomsList2[j])
    #Lcheck.append(["16",len(parsed_dataout)])

    parsed_data2=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    # Convert the string representation of a list to an actual list
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    # Fallback to raw string if parsing fails
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data2:
                parsed_data2.append(parsed_row)
    
    roomsList=["","","","","","","","","","","","","","",""]
    roomsList2=["","","","","","","","","","","","","","",""]
    roomsList3=["","","","","","","","","","","","","","",""]
    roomsList4=["","","","","","","","","","","","","","",""]
    roomsList5=["","","","","","","","","","","","","","",""]
    roomsList6=["","","","","","","","","","","","","","",""]
    roomsListCheck=["kitchen", "bathroom", "living room", "sitting room", "hallway", "garage", "conservatory", "garden room", "dining room", "bedroom", "hallway", "garden", "front of house", "back of house"]

    for j in range(len(parsed_data2)):
        if parsed_data2[j][1].replace(",","").replace(" ","").replace("-","").lower()==parsed_data[i][0][0].replace(",","").replace(" ","").replace("-","").lower():
            for k in range(len(roomsListCheck)):
                if parsed_data2[j][3][0].lower()==roomsListCheck[k]:
                    try:
                        roomsList[k]=parsed_data2[j][3][1]
                        roomsList2[k]=parsed_data2[j][3][2]
                        roomsList3[k]=parsed_data2[j][3][3]
                        roomsList4[k]=parsed_data2[j][3][4]
                        roomsList5[k]=parsed_data2[j][3][5].replace("%","")
                        roomlist6=parsed_data2[3][6].lower()
                        roomlist6=roomlist6.replace("none","0").replace("minor","1").replace("moderate","2").replace("major","3").replace("full","4").replace("extensive","5")
                        roomsList6[k]=roomlist6
                    except:pass
    
    for j in range(len(roomsList)):
        parsed_dataout.append(roomsList[j])
        parsed_dataout.append(roomsList2[j])
        parsed_dataout.append(roomsList3[j])
        parsed_dataout.append(roomsList4[j])
        parsed_dataout.append(roomsList5[j])
        parsed_dataout.append(roomsList6[j])
    
    #Lcheck.append(["17",len(parsed_dataout)])
    parsed_data2=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    # Convert the string representation of a list to an actual list
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    # Fallback to raw string if parsing fails
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data2:
                parsed_data2.append(parsed_row)
    
    roomsList=["","","","","","","","","","","","","","",""]
    roomsList2=["","","","","","","","","","","","","","",""]
    roomsListCheck=["kitchen", "bathroom", "living room", "sitting room", "hallway", "garage", "conservatory", "garden room", "dining room", "bedroom", "hallway", "garden", "front of house", "back of house"]

    for j in range(len(parsed_data2)):
        if parsed_data2[j][0].replace(",","").replace(" ","").replace("-","").lower()==parsed_data[i][0][0].replace(",","").replace(" ","").replace("-","").lower():
            for k in range(len(roomsListCheck)):
                try:
                    if parsed_data2[1][0].lower()==roomsListCheck[k]:
                        roomsList[k]=parsed_data2[1][1].replace("%","")
                        roomlist2=parsed_data2[1][2].lower()
                        roomlist2=roomlist2.replace("none","0").replace("minor","1").replace("moderate","2").replace("major","3").replace("full","4").replace("extensive","5")
                        roomsList2[k]=roomlist2
                except:pass

    for j in range(len(roomsList)):
        parsed_dataout.append(roomsList[j])
        parsed_dataout.append(roomsList2[j])
    Lcheck.append(["18",len(parsed_dataout)])
    new_list = []
    for item in parsed_dataout:
        if isinstance(item, list):
            new_list.extend(item)  
        else:
            new_list.append(item)
    parsed_dataout=new_list
    Lcheck.append(["19",len(parsed_dataout)])
    #if len(parsed_dataout)!=387:
    print(parsed_data[i][0][0])
    print(Lcheck)
    with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile2:
        csvwriter = csv.writer(csvfile2)
        csvwriter.writerow(parsed_dataout)




[['1', 109], ['2', 134], ['3', 134], ['11', 134], ['12', 169], ['12.1', 170], ['12.2', 173], ['12.3', 174], ['12.4', 175], ['12.5', 177], ['12.6', 183], ['12.1', 184], ['12.2', 187], ['12.3', 188], ['12.4', 189], ['12.5', 191], ['12.6', 197], ['13', 197], ['14', 197], ['15', 197], ['16', 253], ['17', 343], ['18', 373], ['19', 401]]
[['1', 109], ['2', 134], ['3', 134], ['11', 134], ['12', 169], ['12.1', 170], ['12.2', 173], ['12.3', 174], ['12.4', 175], ['12.5', 177], ['12.6', 183], ['13', 183], ['14', 183], ['15', 183], ['16', 239], ['17', 329], ['18', 359], ['19', 387]]
[['1', 109], ['2', 134], ['3', 134], ['4', 144], ['5', 146], ['6', 154], ['7', 155], ['8', 163], ['9', 164], ['10', 172], ['11', 173], ['12', 169], ['12.1', 170], ['12.2', 173], ['12.3', 174], ['12.4', 175], ['12.5', 177], ['12.6', 183], ['13', 183], ['14', 183], ['15', 183], ['16', 239], ['17', 329], ['18', 359], ['19', 387]]




