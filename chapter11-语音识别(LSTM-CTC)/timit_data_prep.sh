#!/bin/bash

#此文件用来得到训练集，验证集和测试集的音频路径文件和转录文本即标签文件以便后续处理
#输入的参数时TIMIT数据库的路径。
#更换数据集之后，因为数据的目录结构不一致，需要对此脚本进行简单的修改。

if [ $# -ne 1 ]; then
   echo "Need directory of TIMIT dataset !"
   exit 1;
fi

conf_dir=`pwd`/conf
prepare_dir=`pwd`/data_prepare
log_dir=`pwd`/log

[ -f $conf_dir/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf_dir/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";

#根据数据库train，test的名称修改，有时候下载下来train可能是大写或者是其他形式
train_dir=train
test_dir=test

ls -d "$*"/$train_dir/dr*/* | sed -e "s:^.*/::" > $conf_dir/train_spk.list

tmpdir=`pwd`/tmp
mkdir -p $tmpdir $prepare_dir $log_dir
for x in train dev test; do
  # 只使用 si & sx 的语音.
  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $conf_dir/${x}_spk.list > $tmpdir/${x}_sph.flist

  #获得每句话的id标识
  sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' $tmpdir/${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
  
  #生成wav.scp,即每句话的音频路径
  paste $tmpdir/${x}_sph.uttids $tmpdir/${x}_sph.flist \
    | sort -k1,1 > $prepare_dir/${x}_wav.scp

  #把.wrd中的每句文本标签放到一个文件中进行后续处理
  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.wrd' \
    | grep -f $conf_dir/${x}_spk.list > $tmpdir/${x}_txt.flist
  sed -e 's:.*/\(.*\)/\(.*\).wrd$:\1_\2:i' $tmpdir/${x}_txt.flist \
    > $tmpdir/${x}_txt.uttids
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    cut -f3 -d' ' "$line" | tr '\n' ' ' | sed -e 's: *$:\n:'
  done < $tmpdir/${x}_txt.flist > $tmpdir/${x}_txt.trans
  
  #将句子标识（uttid）和文本标签放在一行并按照uttid进行排序使其与音频路径顺序一致
  paste $tmpdir/${x}_txt.uttids $tmpdir/${x}_txt.trans \
    | sort -k1,1 > $tmpdir/${x}.trans
  
  #生成文本标签
  cat $tmpdir/${x}.trans | sort > $prepare_dir/$x.text || exit 1;
done

rm -rf $tmpdir

echo "Data preparation succeeded"
