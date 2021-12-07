i=0

for file in ~/Documents/ner/ner/huggingface/*_cross*/*.pt
do
    name=`echo $(basename "$file" .pt) | rev | cut -c2- | rev`
    echo "${name}${i}"
    cp -i "$file" "ner_models/${name}${i}.pt"
    ((i=i+1))
done
