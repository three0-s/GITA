# @author: Yewon Lim (ga060332@yonsei.ac.kr)
def print_table(col_names, *cols, tab_width=50, just='right'):
    '''Make and return a string of a table that consists of multiple columns.
    
    The length of 'col_names' should be the same as of 'cols'.

    Params]
        col_names: A list of column names
        cols: Vector like 1D variables, each of which will be a column of the table sequentially
        tab_width (optional): An width of a table. default = 50
        just (optional): Justification option. 'center' and 'right' are acceptible. default = 'right'
    '''
    # Assume that len(col_names) == len(cols) holds.
    # Column names
    return_str=''
    return_str += ("="*tab_width+'\n')
    cols_str = ''.join([str(name).center(int((tab_width//len(col_names))*0.95)) 
                        for name in col_names])
    return_str += (cols_str+'\n')
    return_str += ('-'*tab_width+'\n')
    
    # get the maximum length of contents, respect to each column
    max_data_len = []
    for col in cols:
        max_len = 0
        for i in range(len(col)):
            if max_len < len(str(col[i])):
                max_len = len(str(col[i]))
        max_data_len.append(max_len)

    # print data row by row, propotionally indented to the length of each column
    for row in range(len(cols[0])):
        # right justification
        if just =='right':
            row_str = ''.join([str(data[row]).rjust(max_data_len[i]).center(tab_width // len(cols))
                          for i, data in enumerate(cols)])
        # center justification
        elif just =='center':
            row_str = ''.join([str(data[row]).center(max_data_len[i]).center(tab_width // len(cols))
                          for i, data in enumerate(cols)])
        return_str+=(row_str+'\n')
    
    return_str+=('='*tab_width+'\n')
    return return_str
