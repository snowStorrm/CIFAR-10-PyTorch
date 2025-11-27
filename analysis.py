import pandas, openpyxl

# Simply exports two arrays to an excel sheet for data analysis
# - Will append to an existing excel file if it exists
def exportResults(axisX, axisY, name=f"Sheet{len(openpyxl.load_workbook('./out.xlsx').sheetnames)+1}"):
    a = []
    for i in range(len(axisX)):
        a.append(' ')
    dataToAppend = pandas.DataFrame([axisX, axisY, a], columns=['x', 'y', ' '])
    with pandas.ExcelFile(path_or_buffer='./out.xlsx') as reader:
        data = reader.parse(sheet_name=name)
    with pandas.ExcelWriter(path='./out.xlsx', mode='a', engine='openpyxl', if_sheet_exists="overlay") as writer:
        pandas.DataFrame(pandas.concat([data, dataToAppend], axis=1)).to_excel(excel_writer=writer, sheet_name=name)
        writer.close()
        