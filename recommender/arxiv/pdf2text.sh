pdf_dir="$1";
txt_dir="$2";

for full_file in "$pdf_dir/"*.pdf; do 
	dir_name=$(dirname "$full_file");
	file_name=$(basename "$full_file");
	file_name="${file_name%.*}";
	echo "Converting $file_name...";
	pdftotext "$full_file" "$txt_dir/$file_name.txt"; 
done