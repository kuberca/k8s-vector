cat $1 | tr -s ' ' | cut -d ' ' -f 8- > $1.rmtime
