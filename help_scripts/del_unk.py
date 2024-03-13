def clean_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            cleaned_line = ' '.join(word for word in line.split() if word != '<unk>')
            f_out.write(cleaned_line + '\n')

input_file = '/home/maryna/HSE/DL/Translator/beam_4_answer_8_cycle_lr_big_model.txt'
output_file = '/home/maryna/HSE/DL/Translator/beam_4_answer_8_cycle_lr_big_model_no_unk.txt'

clean_file(input_file, output_file)
