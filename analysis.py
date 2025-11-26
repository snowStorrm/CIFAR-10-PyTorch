import pandas, openpyxl

# Simply exports two arrays to an excel sheet for data analysis
# - Will append to an existing excel file if it exists
def exportResults(axisX, axisY, name=f"Sheet{len(openpyxl.load_workbook('./out.xlsx').sheetnames)}"):
    a = []
    for i in range(len(axisX)):
        a.append(' ')
    with pandas.ExcelWriter(path='./out.xlsx', mode='a', engine='openpyxl', if_sheet_exists="new") as writer:
        pandas.DataFrame({"x":axisX, "y":axisY, " ": a}).to_excel(excel_writer=writer, sheet_name=f"Sheet {name}")
        