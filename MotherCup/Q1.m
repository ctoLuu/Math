file = fopen('file1.csv');
csvData = readtable('file1.csv');
opts = detectImportOptions('data.csv', 'VariableNamingRule', 'preserve');
categories = csvData{:,1}; % 提取第一列数据，即分拣中心的类别
dates = csvData{:,2}; % 提取第二列数据，即日期
cargoVolumes = csvData{:,3}; % 提取第三列数据，即货运量
dates = datetime(dates); % 假设CSV中的日期格式为'年/月/日'
matrixData = [categories; dates; cargoVolumes];
matrix = [C{1}, C{2}, C{3}];
fclose(file);

