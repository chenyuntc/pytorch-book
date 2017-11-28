这是书籍

##知识图谱

## 说明


notebook: notebook内容是书中的草稿,为了更加方便读者阅读,并不把内容
- 错别字,描述不通顺的
如果想阅读更完善的内容,请书中,书中内容经过多次校对,已将错别字,表述不通顺的内容降到最低.


还有不少的错误,书中的内容,相对于这里的内容
修复了





```Python
visdom
torchvision
tensorboard_logger
fire
ipdb
IPython
Jupyter
```


## 关于Visdom 问题及其解决：

不知道从什么时候起，visdom 就不能用，经过分析发现是两个js被防火墙给阻挡了：
- `https://cdn.rawgit.com/plotly/plotly.js/master/dist/plotly.min.js`
- `https://cdn.rawgit.com/STRML/react-grid-layout/0.14.0/dist/react-grid-layout.min.js`

最简单的解决方法
- step1 找到visdom的`index.html`文件
```Bash
locate visdom/static/index.html
```
输出
```
/usr/local/lib/python2.7/dist-packages/visdom/static/index.html
/usr/local/lib/python3.5/dist-packages/visdom/static/index.html
```
- step2:下载[我修改过后的文件](http://pytorch-1252820389.cosbj.myqcloud.com/visdom/index.html)，替换这两个index.html,可能需要root权限。


其它的解决方法，包括：
- 下载这两个文件到本地，然后修改index.html中都应js文件的路径
- 使用代理，但是把某些域名加入白名单
- ....
