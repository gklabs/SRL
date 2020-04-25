#! /bin/tcsh

# sections that are considered to generate training data; section numbers should be sorted 
set SECTIONS = "02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21"


# if you feel that 4 sections is enough training data, use the following 
# set SECTIONS = "15 16 17 18"

# name of the output file 
set FILE = "train-set" 
set s = "IDK"
#foreach s ( $SECTIONS )

echo Processing section $s

cat words/test.wsj.words.train > /tmp/$$.words
cat props/test.wsj.props.train > /tmp/$$.props
    
## Coose syntax
cat synt.col2/test.wsj.synt.col2.train > /tmp/$$.synt
cat synt.col2h/test.wsj.synt.col2h.gz.train > /tmp/$$.synt
cat synt.upc/test.wsj.synt.upc.train > /tmp/$$.synt
cat synt.cha/test.wsj.synt.cha.train > /tmp/$$.synt

cat senses/test.wsj.senses.train > /tmp/$$.senses
cat ne/test.wsj.ne.train > /tmp/$$.ne


paste -d ' ' /tmp/$$.words /tmp/$$.synt /tmp/$$.ne /tmp/$$.props | gzip > /tmp/$$.section.$s.gz
#end

echo Generating gzipped file training.gz
zcat /tmp/$$.section* | gzip -c > training.txt.gz

echo Cleaning files
rm -f /tmp/$$*

