# 字典是非序列的資料結構
# 無法使用類似鑽列的索引觀念取得元素內容

# 他的元素採用 鍵:值 的方式配對
# 操作時用KEY取得VALUE的內容

# 實際運用中，我們查詢KEY，就可以列印出對應的值內容
# 最右邊如果再加1個, 也不會影響

name_dict = {'key1': 'value1', 
             'key2': 'value2',
             'key3': 'value3',}


#%%
# 實例
fruits = {'西瓜':15, '香蕉':20, '水蜜桃':25}
noodles = {'牛肉麵':100, '肉絲麵': 80, '陽春麵':60}
print(fruits)
print(noodles)
# %%

# 定義遊戲腳色
soldier0 = {'tag':'red', 'score':3}
print(soldier0)
# %%
# 列出字典元素的值
# 用字典變數['鍵']取得值
print(fruits['水蜜桃'])
print("水蜜桃一斤 =", fruits['水蜜桃'], '元')
print("牛肉麵一碗 =", noodles['牛肉麵'], '元')
# %%

# 分別列出小兵字典的tag和score的值
soldier0 = {'tag':'red', 'score':3}
print("你剛打死標記 %s 小兵" % soldier0['tag'])
print('可以得到', soldier0['score'], "分")
# %%
# 有趣的列出特定鍵的值
fruits = {0:'西瓜', 1:'香蕉', 2:'水蜜桃'}
print(fruits[0], fruits[1], fruits[2])
# %%

# 增加字典元素
# name_dict['鍵'] = '值'
# name_dict是字典變數

fruits = {'西瓜':15, '香蕉': 20, '水蜜桃':25}
fruits['橘子'] = 18
print(fruits)
print('橘子一斤 = ', fruits['橘子'], '元')

print('你剛剛付了 %s 元' % fruits['橘子'])



# %%
#為soldier0字典增加x,y座標(xpos, ypos)和移動速度(speed)元素
#同時列出結果作驗證

soldier0 = {'tag':'red', 'score':3}
soldier0['xpos'] = 100
soldier0['ypos'] = 30
soldier0['speed'] = 'slow'
print('小兵的x座標 =', soldier0['xpos'])
print('小兵的y座標 =', soldier0['ypos'])
print('小兵的移動速度 = ', soldier0['speed'])
# %%
# 更改元素內容
fruits = {'西瓜':15, '香蕉':20, '水蜜桃': 25}
print('舊價格香蕉一斤= ', fruits['香蕉'], '元')
fruits['香蕉'] = 12
print('新價格香蕉一斤= ', fruits['香蕉'], '元')
# %%
# 我們需要移動小兵的位置
