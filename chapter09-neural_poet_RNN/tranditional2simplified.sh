for fff in `ls *.json`
do
cconv -f utf8-tw  -t UTF8-CN $fff  -o simplified/$fff
done
