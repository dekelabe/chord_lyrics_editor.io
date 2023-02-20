
import tkinter as tk
from tkinter import messagebox,filedialog
import pyperclip
import unicodedata
from keras.models import load_model
import tensorflow as tf
import numpy as np

#imports from the custom chord predictor libarary: 
from jl_dictionaries import Dictionaries
from jl_encoding import Encoder


Dictionaries.get_all_possible_chords()
vocab_size = len(Dictionaries.get_all_possible_chords())
vocab_size
encoder = Encoder(Dictionaries.get_all_possible_chords())


#this takes time so i do it after the gui is opened
model = load_model('lstm_normalised__W_20_lr_0_0005_epochs=50_batch_128.h5')

basic=['A','Bb','B','C','C#','D','Eb','E','F','F#','G','Ab']
minors=[b+'m' for b in basic]
basic_7s=[b+'7' for b in basic]
basic_m7s=[b+'m7' for b in basic]
basic_maj7s=[b+'maj7' for b in basic]
basic_m7b5=[b+'m7b5' for b in basic]
basic_dim7=[b+'dim7' for b in basic]
basic_4=[b+'4' for b in basic]
basic_add9=[b+'add9' for b in basic]

allchords=[basic,minors,basic_7s,basic_m7s,basic_maj7s,basic_m7b5,basic_dim7,basic_4,basic_add9]
print(allchords)

def init_chords_options(available_chords_list_option):
    global chords_options_list
    operations_list=['delete','copy_row','paste_row','copy_verse','paste_verse']

    common_simple_chords=['A','B','C','D','E','F','G']
    common_simple_chords=[[c+'',c+'m',c+'#',c+'b',c+'#m',c+'7',c+'bm'] for c in common_simple_chords]
    common_simple_chords =[chord for chord_types in common_simple_chords for chord in chord_types if chord in encoder.number_to_category_dict.values()]    
    if available_chords_list_option=='restricted':   
        #define what chords we wish to work with: (there are 804 options, but we only work with basic ones)
        chords_options_list = common_simple_chords
        #temp - remove chords that are not in the chords list: 
        print(chords_options_list)
    elif available_chords_list_option=='common':
        chords_options_list=list(encoder.number_to_category_dict.values())
        #get rid of rare chords (such as those containing 13s, 6s, and so on: )
        chords_ids_to_remove=['13','+','6','sus','add','#9','#11','3','7-9','9-5','11','AmMaj','7-5','aug']
        chords_options_list=[chord for chord in list(encoder.number_to_category_dict.values()) if not any(substring in chord for substring in chords_ids_to_remove)]
        #put the more common chords first in the list:
        # first, remove them from the entire list:  
        for chord in common_simple_chords: 
            chords_options_list.remove(chord)
        #then concatinate them (the common ones are first now) 
        chords_options_list=operations_list+common_simple_chords+chords_options_list 
    elif available_chords_list_option=='all':
        chords_options_list=list(encoder.number_to_category_dict.values())
        #put the more common chords first in the list:
        # first, remove them from the entire list:  
        for chord in common_simple_chords: 
            chords_options_list.remove(chord)
        #then concatinate them (the common ones are first now) 
        chords_options_list=operations_list+common_simple_chords+chords_options_list    
    else: 
        Exception('type method is non valid (must be either "all" or "restricted"')
    return chords_options_list

def pad_sequence(input, length):
        return [0] * (length - len(input)) + input

def get_predictions(model,prev_chords,allowed_chords,top_n=10):
    sequence=[encoder.to_number(chord) for chord in prev_chords if (chord in encoder.category_to_number_dict.keys())]
    if len(sequence)>20:  #if we allready have more than 20 chords, just take the most recent 20 chords
        padded_sequence=sequence[-20:]
    else: 
        padded_sequence=pad_sequence(sequence,20)

    predicted_p_unsorted=model(np.array([padded_sequence]))
    predicted_p_unsorted_array=np.array(predicted_p_unsorted)
    descending_sorting_inds=np.argsort(-1*predicted_p_unsorted).flatten() #negative so sorting will result in descending order
    sorted_predicted_ps=(np.sort(-1*predicted_p_unsorted)*-1).flatten()

    allowed_chords_in_order=[ind for ind in descending_sorting_inds if (not(ind==0) and (encoder.number_to_category_dict[ind] in allowed_chords))]
    allowed_chords_proba_in_order=[predicted_p_unsorted_array[0][ind] for ind in descending_sorting_inds if (not(ind==0) and (encoder.number_to_category_dict[ind] in allowed_chords))]

    top_n_chords=[encoder.number_to_category_dict[chord] for chord in allowed_chords_in_order[0:top_n]]
    top_n_chords_p=allowed_chords_proba_in_order[0:top_n]
    return top_n_chords,top_n_chords_p

#top_n_chords,top_n_chords_p=get_predictions(model=model,prev_chords=['C','F','F#','A','G','A','G']*6,allowed_chords=['G','F#','C'],top_n=2)
#print('predicted chords:',top_n_chords)
#print('predicted probabilities:',top_n_chords_p)

copied_line=[]
added_chords=[]
empty_chord_space_symbol='_'
max_words_in_sentence=8
how_many_cords_before_resorting_list=1
hebrew_flag=False
start=[]
end=[]
copied_verse_in_lines=[]
copied_verse_start=[]
copied_verse_end=[]

chords_options_list=init_chords_options('common')

def is_hebrew(term):
    return 'HEBREW' in unicodedata.name(term.strip()[0])


def get_ordered_added_chords_from_text(curr_text_string,empty_chord_space_symbol):
    lines=curr_text_string.split('\n')
    added_cords_from_text=[]
    for line in lines: 
        if line.count(empty_chord_space_symbol)>3:
            line_chords=[c for c in line.split(empty_chord_space_symbol) if len(c)>0]
            if hebrew_flag: 
                line_chords=line_chords[::-1]
            added_cords_from_text=added_cords_from_text+line_chords
    return  added_cords_from_text  


def on_double_click(event,widget):
    global chords_options_list,added_chords,copied_line,start,end,copied_verse_in_lines,copied_verse_start,copied_verse_end
    how_many_cords_before_resorting_list=5
    if widget=='chords_list':
        curr_chord = chords_list.get(chords_list.curselection())
    elif widget=='suggested_chords_list':
        curr_chord = suggested_chords_list.get(suggested_chords_list.curselection())
        curr_chord=curr_chord.split(' ')[0]
    elif widget=='chord_table':
        print('selection from chord_table')
        curr_chord=event.widget.cget("text")

    curr_text_string=output_text.get('1.0',tk.END)

    added_chords=get_ordered_added_chords_from_text(curr_text_string,empty_chord_space_symbol) #this line might be useless as we call it again at the end of the function
    chord_length = len(curr_chord)

    print('selected chord:',curr_chord)
    print('current state of chords list added by user',added_chords)
    print('last chord length in list:',chord_length)
    #print(f'selected chord {chord} length {chord_length}')

    #get the chord left to the cursor by looking at the text string: 
    line_num, col_num = output_text.index('insert').split('.')
    print('selection location on the grid',(line_num, col_num))
    text_string_lines=curr_text_string.split('\n')
    line_num=int(line_num)-1
    col_num=int((int(col_num)-1))
    print('location_on_string:',(line_num,col_num))
    text_string_lines[line_num][col_num]
    print('selected text line string',text_string_lines[line_num])
    for m in np.arange(col_num,-1,-1):
        if text_string_lines[line_num][m]==empty_chord_space_symbol:
            break
    chord_left_to_selection_name_from_string=text_string_lines[line_num][m+1:col_num+1]
    print('chord left of cursor: ',chord_left_to_selection_name_from_string)

    #delete the previous chord from the same row it was inserted at
    if curr_chord=='delete': #delete the chord on the left: 
        chord_length=len(chord_left_to_selection_name_from_string)
        output_text.delete("insert -{}c".format(chord_length), "insert")
        print('state of chord list before deleting: ',added_chords) 
        print('last chord length=',chord_length)
        output_text.insert(tk.INSERT, empty_chord_space_symbol*chord_length)
    elif curr_chord=='copy_row':
        output_text.tag_remove("highlight", 1.0, "end")
        copied_line=text_string_lines[line_num]
        line_len=len(copied_line)
        print('copied the current chord row:',copied_line)
        output_text.tag_add("highlight",str(line_num+1)+'.0',str(line_num+1)+'.'+str(line_len))
    elif curr_chord=='paste_row':
        if len(copied_line)>0:
            curr_line=text_string_lines[line_num]
            line_len=len(curr_line)
            output_text.delete(str(line_num+1)+'.0',str(line_num+1)+'.'+str(line_len))
            output_text.insert(str(line_num+1)+'.0', copied_line)
    elif curr_chord=='copy_verse':
        #get the end verse location ()
        copied_verse_in_lines=extract_chords_from_verse(curr_text_string,start,end)
        copied_verse_start=int(start.split('.')[0])
        copied_verse_end=int(end.split('.')[0])
        output_text.tag_remove("highlight", 1.0, "end")
        copied_line=text_string_lines[line_num]
        line_len=len(copied_line)
        print('copied the current verse:',copied_verse_in_lines)
        output_text.tag_add("highlight",start.split('.')[0]+'.0',end.split('.')[0]+'.'+str(line_len))
    elif curr_chord=='paste_verse':
        if len(copied_verse_in_lines)>0:
            current_cursor_line=line_num
            copied_text_num_lines=len(copied_verse_in_lines)
            temp_text_lines=text_string_lines[line_num:line_num+copied_text_num_lines]
            line_len=len(temp_text_lines[0])
            #take the chord lines from the copied_verse_lines: 
            temp_text_lines[0::2]=copied_verse_in_lines[0::2]
            temp_text_lines_string='\n'.join(temp_text_lines)
            output_text.delete(str(line_num+1)+'.0',str(line_num+copied_text_num_lines)+'.'+str(line_len))
            output_text.insert(str(line_num+1)+'.0', temp_text_lines_string)       

            #get the chords in that line: 
            temp_line_split=copied_line.split(empty_chord_space_symbol)
            chords_in_line=[c for c in temp_line_split if len(c)>0]
            #add the chords: note that order is not currently maintained 
    else:
        #replacement of the chord to the left: 
        if len(chord_left_to_selection_name_from_string)>0: #altough the user didnt ask for delete operation, the cursor is placed right next to another chord so we will replace it
            
            chord_length=len(chord_left_to_selection_name_from_string)
            print(f'replacing {chord_left_to_selection_name_from_string} with {curr_chord}')
            curr_chord=empty_chord_space_symbol*(len(chord_left_to_selection_name_from_string)-len(curr_chord))+curr_chord
            print('after adding markers:',curr_chord)
            chord_length=max(chord_length,len(curr_chord))
        #regular addition of chord:
        output_text.delete("insert -{}c".format(chord_length), "insert")
        output_text.insert(tk.INSERT, curr_chord)
    
    added_chords=get_ordered_added_chords_from_text(curr_text_string,empty_chord_space_symbol)

    
    print('reordering suggested chords list')
    top_n_chords,top_n_chords_p=get_predictions(model=model,prev_chords=added_chords,allowed_chords=chords_options_list,top_n=15)
    print('predicted chords',top_n_chords)
    suggested_chords_list.delete(0,len(chords_options_list))
    chords_options_list1=top_n_chords
    for curr_chord,chord_p in zip(chords_options_list1,top_n_chords_p):
        insertion_chord_and_probability=f'{curr_chord} - {chord_p:.3f}%'
        suggested_chords_list.insert(tk.END, insertion_chord_and_probability)
            
        

def on_paste_and_parse():
    global hebrew_flag
    text = pyperclip.paste()
    parsed_text = ""
    lines = text.split("\n")
    if len(lines)==1:
        words = text.split(" ")
        parsed_w_text = ""
        for w_cnt,word in enumerate(words):
            parsed_w_text = parsed_w_text +' '+ word
            if np.mod(w_cnt+1,max_words_in_sentence)==0:
                parsed_w_text=parsed_w_text+'\n' 
        lines = parsed_w_text.split("\n")

    largest_string=max(lines,key=len)
    #print('largest string:',largest_string)
    largest_string=len(largest_string)
    words_on_longest_line=lines[lines.index(max(lines,key=len))].split(' ')
    max_num_words_in_sentence=len(words_on_longest_line)
    #print('max words in longest setence:',type(max_num_words_in_sentence))
    
    #check if the language is hebrew: 
    hebrew_flag=any([is_hebrew(word) for word in words_on_longest_line if len(word)>1])
    if hebrew_flag: 
        print('detected hebrew')
        hebrew_flag
    else: 
        print('hebrew not detected')
    
    empty_places_for_chords=largest_string+max_num_words_in_sentence
    parsed_text=empty_places_for_chords*empty_chord_space_symbol + '\n'
    for line in lines:
        if len(line)==0: #empty char, add a whole empty string symbols:
            line=empty_places_for_chords*empty_chord_space_symbol + '\n'
        if hebrew_flag:
            parsed_text += empty_places_for_chords*empty_chord_space_symbol + '\n' + line + "\n"
        else:
            parsed_text += line + "\n" + empty_places_for_chords*empty_chord_space_symbol + '\n'

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, parsed_text)


def save_text():
    edited_text = output_text.get("1.0", "end") # Get the text from the text entry widget
    if not empty_chord_space_symbol==' ':
        edited_text=edited_text.replace(empty_chord_space_symbol,' ')
    print(edited_text)
    
    # Use a dialog box to get the file name to save to
    file_name = filedialog.asksaveasfilename(defaultextension=".txt",initialdir = ".\chord sheets outputs",filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])

    if file_name:
        with open(file_name, "w") as f:
            f.write(edited_text) # Write the text to a file
        print("Text saved successfully.")
    else:
        print("Save operation cancelled.")


root = tk.Tk()
root.title("Guitar Chords")
width  = int(root.winfo_screenwidth()/2)
height = int(root.winfo_screenheight()*0.85)
root.geometry(f'{width}x{height}+10+{int(height*0.025)}')


#######one chord list########
chords_frame = tk.Frame(root)
chords_frame.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(chords_frame, orient="vertical")
scrollbar.pack(side="left", fill="y")

chords_label=tk.Label(chords_frame,text='chords\nlist',height=2, width=8 , fg='black')
chords_label.configure(font=("TkFixedFont", 12, "bold"))
#sugested_chords=tk.Label(chords_frame,text='suggested\nchords',height=2, width=9)
#sugested_chords.configure(font=("TkFixedFont", 12, "bold"))
chords_list = tk.Listbox(chords_frame, xscrollcommand=scrollbar.set, height=1, width=10,bg='lavender',cursor="plus",highlightcolor='red',selectmode='SINGLE')

for chord in chords_options_list:
    chords_list.insert(tk.END, chord)
    
chords_label.pack(side='top',anchor='nw')
chords_list.pack(side="left", fill="both", expand=True)

#sugested_chords.pack(side='top',anchor='nw')
scrollbar.config(command=chords_list.yview)

######second chord list#########
chords_frame1 = tk.Frame(root)
chords_frame1.pack(side="left", fill="both", expand=True)
scrollbar1 = tk.Scrollbar(chords_frame1, orient="vertical")
#scrollbar1.pack(side="left", fill="y")

chords_label1 = tk.Label(chords_frame1,text='suggested\nchords',height=2, width=8 , fg='black')
chords_label1.configure(font=("TkFixedFont", 12, "bold"))
suggested_chords_list = tk.Listbox(chords_frame1, xscrollcommand=scrollbar1.set, height=1, width=15,bg='lavender blush',cursor="arrow")
for chord in []:
    suggested_chords_list.insert(tk.END, chord)

chords_label1.pack(side='top',anchor='nw')
suggested_chords_list.pack(side="left", fill="both", expand=True)
scrollbar1.config(command=suggested_chords_list.yview)
chords_list_num_selected=np.zeros([len(chords_options_list)])

chords_list.bind("<Double-Button-1>", lambda event: on_double_click(event, "chords_list"))
suggested_chords_list.bind("<Double-Button-1>",  lambda event: on_double_click(event, "suggested_chords_list"))

output_frame = tk.Frame(root)
output_frame.pack(side="bottom", fill="both", expand=True)

output_text = tk.Text(output_frame, font=("TkFixedFont", 14),cursor="arrow",undo=True,autoseparators=True,maxundo=-1)

#these 3 functions are built to give the start and end location of the current max text selection
def start_selection(event):
    global start
    start=output_text.index("@%d,%d" % (event.x, event.y))

def continue_selection(event):
    global start,end
    end = output_text.index("@%d,%d" % (event.x, event.y))

def end_selection(event):
    global start,end
    if len(end)==0:
        end=start
    temp=[start,end]
    print('selection start-end: ',(start,end))

def extract_chords_from_verse(curr_text_string,start,end):
    print('entered_verse_parser:')
    start_line_ind=int(start.split('.')[0])-1
    end_line_ind=int(end.split('.')[0])
    curr_text_string_lines=curr_text_string.split('\n')
    copied_verse_in_lines=curr_text_string_lines[start_line_ind:end_line_ind]
    return copied_verse_in_lines

output_text.bind("<Button-1>", start_selection)
output_text.bind("<B1-Motion>", continue_selection)
output_text.bind("<ButtonRelease-1>", end_selection)
output_text.tag_configure("highlight", background="yellow")

output_text.pack(side="top", fill="both", expand=True)

label = tk.Label(root,text = "Lyrics")
label.pack(side="top")

paste_and_parse_button = tk.Button(output_frame, text="Paste and Parse", command=on_paste_and_parse)
paste_and_parse_button.pack(side="bottom",fill="both", expand=False)

save_button = tk.Button(output_frame, text="Save", command=save_text)
save_button.pack(side="bottom", fill="both", expand=False)





root2 = tk.Tk()
root2.title('Common Chords List')
screen_width = root2.winfo_screenwidth()
screen_height = root2.winfo_screenheight()
width = 600
height = 300
x = int(screen_width/2)+10
y = int((screen_height - height)/2)
root2.geometry(f'{width}x{height}+{x}+{y}')

# create a 4x4 table of labels
for i in range(len(allchords)):
    for j in range(len(allchords[0])):
        color = "#%02x%02x%02x" % (int(255-(i*64/11)), int(255-(i*64/11)), int(255-(j*64/11)))
        label = tk.Label(root2, bg=color,text=allchords[i][j], relief="solid", height=1, width=10)
        label.grid(row=i, column=j,padx=3, pady=3)
        label.bind("<Button-1>", lambda event: on_double_click(event, "chord_table"))

# configure the grid to have the same size for each element
for i in range(len(allchords)):
    root2.rowconfigure(i, weight=1)
    for j in range(len(allchords[0])):
        root2.columnconfigure(j, weight=1)

def close_windows():
    root.quit
    root2.quit
    root.destroy()
    root2.destroy()

root.protocol("WM_DELETE_WINDOW", close_windows)
root2.protocol("WM_DELETE_WINDOW", close_windows)
root2.focus_force()
root.focus_force()


root.mainloop()
root2.mainloop()
