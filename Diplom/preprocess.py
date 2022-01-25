import pandas as pd
from datetime import datetime
import re
import category_encoders as ce

def age(df):
  date_dict = {'январь':'january','февраль':'february','март':'march',	
          	'апрель':'april','май': 'may','июнь':	'june',
          	'июль':'july','август':'august','сентябрь':'september',
	          'октябрь':'october','ноябрь':'november','декабрь':'december'} 
  df.age =  df.age.apply(lambda x: date_dict[x.split()[0]] + '/' + \
                         x.split()[1] if x==x else x)
  df.age = df.age.apply(lambda x:int(round((datetime.today() - \
          datetime.strptime(x,'%B/%Y')).days/30,0)) if x == x else x) 
  
def to_number(df,cols):
  for c in cols:
      df[c[0]] = df[c[0]].apply(lambda x: x.split()[0] if x==x else c[1] )
      df[c[0]] = pd.to_numeric(df[c[0]])

def to_bin(df,cols):
  for c in cols:
    df[c] = df[c].apply(lambda x: 1 if x == x else 0)

def m2_regex(x):
    if x == x:
      if len(x) > 0:
        return int(x[0][0])
      else:
        return 0
    else:
      return 0   

def smart_fillna(df,inplace_col,example_col):
  for c in list(df[example_col].unique()):
    try:
      df[inplace_col].mask((df[inplace_col].isna() & (df[example_col] == c)),
                        df[inplace_col][df[example_col] == c].mode()[0],
                        inplace=True)
    except:
      pass  

def extract_dimensions(df):
  
  df['height'] = df.dimensions.str.findall(r'\d+').apply(
                              lambda x: int(x[0]) if x == x else x)
  df['width'] = df.dimensions.str.findall(r'\d+').apply(
                              lambda x: int(x[1]) if x == x else x)
  try:
    df['deep'] = df.dimensions.str.findall(r'\d+').apply(
                              lambda x: int(x[2]) if x == x else x)
  except:
    pass  
  df.drop('dimensions',axis = 1, inplace = True )
  
def full_diameter(vent):
  out = 0
  if vent == vent:
    for x in vent:
      out += int(re.split(r'\D',x)[0]) * int(re.split(r'\D',x)[1])
  return out

proc = pd.read_csv('data/raw_data/proc.csv')
proc = proc[['id','name', 'price', 'link','overclock', 'Серия', 'Кодовое название', 'Разъем (Socket)',
       'Техпроцесс', 'Кол-во ядер', 'Кол-во потоков', 'Тактовая частота',
       'Частота TurboBoost / TurboCore', '3-го уровня L3', 'Модель IGP',
       'Тепловыделение (TDP)', 'Макс. рабочая температура', 'Макс. объем',
       'Макс. частота DDR3', 'Макс. частота DDR4', 'Число каналов',
       'Дата добавления на E-Katalog']]
proc.columns = ['id','name', 'price', 'link','overclock', 'series', 'code_name',
                      'socket','techprocess', 'cores', 'threads', 'freq',
                      'turbo_freq', 'cashe_l3', 'igp','tdp','max_temp',
                      'max_ram','ddr3', 'ddr4', 'channels','age']
proc = proc[~proc.ddr3.isna() | ~proc.ddr4.isna()]
proc.dropna(subset=['techprocess',
                    'threads','freq'],inplace=True)
proc.turbo_freq.fillna(proc.freq,inplace=True)
proc.overclock = proc.overclock.fillna(0).apply(int)
age(proc)
proc.igp.fillna('отсутствует',inplace = True)

"""I've decided to leave NaN's in some cases, 
to choose what to do with them during model training."""

to_number(proc,[['techprocess'], ['cores',None,], ['threads'], ['freq'],
                ['turbo_freq'],['cashe_l3',0], ['tdp',None],['max_temp',None],
                ['max_ram',None], ['ddr3',0],['ddr4',0], ['channels',None]
                ]) 

proc['ram_max_freq'] = proc.ddr3.combine(proc.ddr4,max)
proc.ddr3 = proc.ddr3.apply(lambda x: 1 if x!=0 else 0)
proc.ddr4 = proc.ddr4.apply(lambda x: 1 if x!=0 else 0)
proc.to_csv('data/datasets/proc.csv',index = False)

mb = pd.read_csv('data/raw_data/mb.csv')
try:
  mb = mb[mb.DDR2.isna() &
          mb['IDE разъем'].isna() &
          mb['Модель встроенного процессора'].isna()]
except:
  pass

mb.dropna(subset = ['Socket','Чипсет*'],inplace=True)  
mb = mb[['id', 'name', 'price', 'link', 'По направлению',
'Дата добавления на E-Katalog', 'Socket','Форм-фактор','Размеры (ВхШ)',
'Чипсет*','DDR3','DDR4','DDR5','Форм-фактор слота для памяти',
'Режим работы','Максимальная тактовая частота', 'Максимальный объем памяти',
'Выход DVI', 'Звук (каналов)','SATA2 (3 Гбит/с)',
'SATA3 (6 Гбит/с)','M.2 разъем','Интерфейс M.2', 'LAN (RJ-45)','Кол-во LAN-портов',
'Поддержка PCI Express','Слотов PCI-E 1x','Слотов PCI-E 4x','Слотов PCI-E 8x',
'Слотов PCI-E 16x','PCI-слотов','USB 2.0','USB 3.2 gen1','USB 3.2 gen2',
'USB C 3.2 gen1','USB C 3.2 gen2','USB C 3.2 gen2x2','PS/2', 'Фазы питания',
'Основной разъем питания','Питание процессора', 'Разъемов питания кулеров',
'Версия DisplayPort','Интегрированный RAID контроллер','Интерфейс Thunderbolt',
'Разъем Thunderbolt AIC','Синхронизация подсветки','ARGB LED strip','RGB LED strip',
'CPU Fan 4-pin', 'CPU/Water Pump Fan 4-pin','Chassis/Water Pump Fan 4-pin',
'Усилитель', 'Wi-Fi', 'Модель встроенной видеокарты',"Кол-во socket'ов",
'U.2 разъем','Bluetooth','USB 2.0*','USB 3.2 gen1*','USB C 3.2 gen2*',
 'USB 3.2 gen2*','USB C 3.2 gen2x2*','USB C 3.2 gen1*']]

mb.columns =['id', 'name', 'price', 'link', 'direction','age', 'socket',
'form','dimensions', 'chipset','ddr3','ddr4','ddr5','dimm',
'ram_channels','max_ram_freq', 'max_ram_vol','dvi','audio','sata2',
'sata3','m2','m2_interface', 'lan_speed','lan_ports','pci_e_support','pci_e_1x',
'pci_e_4x','pci_e_8x', 'pci_e_16x','pci','usb2','usb_3.2_g1','usb_3.2_g2','usb-c_3.2_g1',
'usb-c_3.2_g2', 'usb-c_3.2_g2x2','ps/2', 'power_phases','power_connector',
'power_processor', 'power_coolers','display_port','raid','thunder_interface',
'thunder_connector','light_sync','argb_led','rgb_led','cpu_fan', 'cpu_water_fan',
'chassis_water_fan','amplifier', 'wifi', 'video_int',"2*socket", 'u_2','bluetooth',
 'USB 2.0*', 'USB 3.2 gen1*', 'USB C 3.2 gen2*', 'USB 3.2 gen2*', 'USB C 3.2 gen2x2*',
 'USB C 3.2 gen1*']

age(mb)

for c in ['socket','power_processor','direction']:
  mb[c] = mb[c].apply(lambda x:x.split(' /')[0] if x == x else x)

smart_fillna(mb,'dimensions','form')

for c in ['ddr3','ddr4','ddr5','thunder_connector','thunder_interface','raid']:
  mb[c].fillna('',inplace=True)

mb['ram_slots'] = mb.ddr3 + mb.ddr4 + mb.ddr5
mb.ram_slots = mb.ram_slots.apply(lambda x: int(x.split()[0]))

for d in ['ddr3','ddr4','ddr5']: 
  mb[d] = mb[d].apply(lambda x: 0 if x == '' else 1)

mb.dimm = mb.dimm.apply(lambda x: 1 if x == 'DIMM' else 0)
mb.ram_channels.replace('2-х/4-х канальный','4',inplace=True)
mb.ram_channels = mb.ram_channels.apply(lambda x: int(x[0]) if x == x else 2 )

to_bin(mb,['dvi','light_sync','amplifier','wifi','video_int','2*socket','bluetooth'])


to_number(mb,[['audio',0],['sata2',0],['sata3',0],['m2',0],
              ['max_ram_freq',None],['max_ram_vol',None],['lan_speed',0],
              ['lan_ports',0],['pci_e_1x',0], ['pci_e_4x',0], ['pci_e_8x',0],
              ['pci_e_16x',0],['pci',0], ['usb2',0], ['usb_3.2_g1',0], 
              ['usb_3.2_g2',0],['usb-c_3.2_g1',0],['usb-c_3.2_g2',0],
              ['usb-c_3.2_g2x2',0],['ps/2',0],['power_phases',None],
              ['power_coolers',0],['rgb_led',0],['argb_led',0],['cpu_fan',0],
              ['cpu_water_fan',0],['chassis_water_fan',0], ['u_2',0],
              ['USB 2.0*',0], ['USB 3.2 gen1*',0], ['USB C 3.2 gen2*',0], 
              ['USB 3.2 gen2*',0],['USB C 3.2 gen2x2*',0],['USB C 3.2 gen1*',0]
              ])

mb['m2_pci_e_2x'] = mb.m2_interface.str.findall(r'\dxPCI-E 2x')
mb.m2_pci_e_2x = mb.m2_pci_e_2x.apply(lambda x:m2_regex(x))
mb['m2_pci_e_4x'] = mb.m2_interface.str.findall(r'\dxPCI-E 4x')
mb.m2_pci_e_4x = mb.m2_pci_e_4x.apply(lambda x:m2_regex(x))
mb['m2_sata_2x'] = mb.m2_interface.str.findall(r'\dxSATA/PCI-E 2x')
mb.m2_sata_2x = mb.m2_sata_2x.apply(lambda x:m2_regex(x))
mb['m2_sata_4x'] = mb.m2_interface.str.findall(r'\dxSATA/PCI-E 4x')
mb.m2_sata_4x = mb.m2_sata_4x.apply(lambda x:m2_regex(x))

mb.pci_e_support = mb.pci_e_support.fillna(0).apply(int)

mb.display_port = mb.display_port.apply(lambda x: 1 if x==x else 0)

mb['raid_0_1'] = mb.raid.apply(lambda x: 1 if x == x else 0)
mb['raid_5'] = mb.raid.apply(lambda x: 1 if '5' in x else 0)
mb['raid_10'] = mb.raid.apply(lambda x: 1 if '10' in x else 0)

mb['thunder_v3'] = (mb.thunder_interface + mb.thunder_connector).apply\
                    (lambda x: int(x.split(' шт')[-2][-1]) if 'v3' in x else 0)
mb['thunder_v4'] = (mb.thunder_interface + mb.thunder_connector).apply\
                    (lambda x: int(x.split(' шт')[-2][-1]) if 'v4' in x else 0)

mb.drop(['thunder_interface','thunder_connector',
         'm2_interface'],axis=1,inplace=True)                                        
extract_dimensions(mb)
mb.to_csv('data/datasets/mb.csv',index = False)

video = pd.read_csv('data/raw_data/video.csv')
video.dropna(subset = ['Модель GPU','Потоковых процессоров'],inplace=True)  

video = video[['id', 'name', 'price', 'link', 'Подключение', 'Модель GPU',
'Объем памяти', 'Тип памяти', 'Разрядность шины','Частота работы GPU',
'Частота работы памяти', 'Техпроцесс', 'Макс. разрешение',
'DisplayPort', 'Версия DisplayPort', 'USB C', 'Версия DirectX',
'Версия OpenGL', 'Потоковых процессоров',
'Макс. подключаемых мониторов', 'Охлаждение', 'Кол-во вентиляторов',
'Потребляемая мощность', 'Дополнительное питание',
'Рекомендуемая мощность БП от', 'Занимаемых слотов', 'Длина видеокарты',
'Дата добавления на E-Katalog', 'Тест Passmark G3D Mark', 'HDMI',
'Текстурных блоков', 'miniDisplayPort',
'Синхронизация подсветки', 'VGA', 'DVI-I',
'DVI-D']]

video.columns = ['id', 'name', 'price', 'link', 'connection', 'gpu', 'mem_vol',
'mem_type', 'bus','gpu_freq','mem_freq','techprocess','max_res', 'displayport',
'displayport_ver', 'usb_c', 'directX','openGL','processors','max_monitors',
'cool', 'vent','power', 'add_power','supply_power','slots', 'length','age',
'passmark','hdmi','text_blocks','miniDisplayPort','light_sync',
'vga', 'dvi-i','dvi-d' ]

age(video)

for c in ['connection']:
  video[c] = video[c].apply(lambda x:x.split(' /')[0] if x == x else x)

for c in ['power','supply_power','passmark','text_blocks',
          'gpu_freq','techprocess']:
  smart_fillna(video,c,'gpu')

smart_fillna(video,'length','slots')


to_number(video,[['mem_vol',0],['bus',None],['gpu_freq',None],['mem_freq',None],
                 ['techprocess',None],['displayport',0],['directX',0], ['vent',0], 
                 ['power',None],['supply_power',None],['length',None],
                 ['passmark',None],['hdmi',0],['miniDisplayPort',0]
                 ])

to_bin(video,['usb_c','light_sync','vga','dvi-i','dvi-d'])

video.max_res = video.max_res.apply(lambda x:int(x.split('x')[0] if x == x else -1) )
video.displayport_ver.fillna('no',inplace=True)
video.openGL.fillna(0,inplace=True)
video.processors = video.processors.apply(int)
video.max_monitors = video.max_monitors.fillna(0).apply(int)
video.cool.fillna('no',inplace=True)
video.add_power.fillna('no',inplace=True)
video.slots.fillna(video.slots.median(),inplace=True)
video.to_csv('data/datasets/video.csv',index = False)


ssd = pd.read_csv('data/raw_data/ssd.csv')
ssd.dropna(subset = ['Форм-фактор','Объем','Тип'],inplace=True)  
ssd = ssd[['id', 'name', 'price', 'link', 'Тип', 'Объем', 'Форм-фактор', 'Разъем',
       'Тип памяти', 'Внешняя скорость записи', 'Внешняя скорость считывания',
       'Ударостойкость при работе', 'Наработка на отказ', 'IOPS записи',
       'IOPS считывания', 'TBW', 'DWPD', 'Гарантия производителя',
       'Дата добавления на E-Katalog', 'Интерфейс M.2', 'Контроллер', 'NVMe',
       'Размеры', 'Назначение']]

ssd.columns = ['id', 'name', 'price', 'link', 'type', 'volume', 'form', 'connector',
       'memory', 'write', 'read','resist', 'mtbf', 'iops_wr',
       'iops_r', 'tbw', 'dwpd', 'warranty','age', 'm.2', 'controller', 'nvme',
       'dimensions', 'server']

age(ssd)

for c in ['connector','memory','m.2',
          'controller']:
  ssd[c] = ssd[c].apply(lambda x:x.split(' /')[0] if x == x else x)

for c in['connector','memory','m.2',
         'controller','nvme']:
  ssd[c].fillna('no', inplace = True)


to_number(ssd,[['volume'],['write',None],['read',None],['resist',None],
               ['mtbf',None],['iops_wr',None],['iops_r',None],['tbw',None],
               ['dwpd',None],['warranty',None]
               ])

extract_dimensions(ssd)
to_bin(ssd,['server'])
ssd.to_csv('data/datasets/ssd.csv',index = False)


ram = pd.read_csv('data/raw_data/ram.csv')
ram.dropna(subset = ['Объем памяти комплекта','Тип памяти','Форм-фактор памяти',
                     'Пропускная способность','CAS-латентность',
                     'Тактовая частота','Рабочее напряжение',
                     'Кол-во планок в комплекте','Дата добавления на E-Katalog'],inplace=True)  

ram = ram[(ram['Тип памяти'] == 'DDR3') | (ram['Тип памяти'] == 'DDR4')]

ram = ram[['id', 'name', 'price', 'link', 'Объем памяти комплекта',
       'Кол-во планок в комплекте', 'Форм-фактор памяти', 'Тип памяти',
       'Тактовая частота', 'Пропускная способность', 'CAS-латентность',
       'Рабочее напряжение', 'Тип охлаждения','Дополнительно',
       'Синхронизация подсветки','Дата добавления на E-Katalog', 'Ранг памяти']]

ram.columns = ['id', 'name', 'price', 'link', 'volume','n_planks', 'dimm',
               'ddr3','freq', 'bandwidth', 'cas','voltage', 'cooling',
               'overclock','lighting','age', 'rank']

age(ram)

to_number(ram,[['volume'],['n_planks'],['freq'],['bandwidth'],
               ['voltage']
               ])

ram.cas = ram.cas.apply(lambda x:int(x.replace('CL','')))

ram.overclock = ram.overclock.apply(
                lambda x: 1 if x==x and ('overclock' in x) else 0)
to_bin(ram,['lighting'])
ram['rank'] = ram['rank'].apply(lambda x:x.split(' /')[0] if x == x else 1)
ram['rank'] = ram['rank'].apply(lambda x:1 if (type(x) == str) and ('одно' in x) else x)
ram['rank'] = ram['rank'].apply(lambda x:2 if (type(x) == str) and ('двух' in x) else x)
ram['rank'] = ram['rank'].apply(lambda x:4 if (type(x) == str) and ('четырех' in x) else x)

ram.dimm = ram.dimm.apply(lambda x: 1 if x == 'DIMM' else 0)
ram.ddr3 = ram.ddr3.apply(lambda x: 1 if x == 'DDR3' else 0)
ram.cooling = ram.cooling.apply(lambda x: 0 if x == 'без охлаждения' else 1)

ram.to_csv('data/datasets/ram.csv',index = False)


power = pd.read_csv('data/raw_data/power.csv')
power = power[['id', 'name', 'price', 'link', 'Мощность', 'Форм-фактор',
       'Система охлаждения', 'Диаметр вентилятора',
       'Сертификат', 'Стандарт ATX 12В v.', 
       'Питание MB/CPU', 'SATA', 'MOLEX','PCI-E 8pin (6+2)', 'Система кабелей',
       'Габариты (ВхШхГ)', 'Дата добавления на E-Katalog',
       'Тип PFC','КПД', 'Тип подшипника', 'Стандарт EPS 12В v.', 
       '+3.3V', '+5V', '+12V1', '-12V', '+5Vsb', 'Мощность +12V',
       'Мощность +3.3V +5V', 'Гарантия производителя']]

power.columns= ['id', 'name', 'price', 'link', 'power', 'form',
       'cool', 'vent','sertificate', 'atx_std', 'mb/cpu', 'sata', 'molex','pci_e',
       'cable_sys','dimensions','age','active_pfc','efficiency', 'bearing', 'eps', 
       '+3.3V', '+5V', '+12V1', '-12V', '+5Vsb', '+12V_watt',
       '+3.3V_+5V_watt', 'warranty']

power.dropna(subset = ['power','form','cool','mb/cpu',
                       'cable_sys'],inplace=True)  
age(power)
power.active_pfc = power.active_pfc.apply(
               lambda x: 1 if x==x and ('актив' in x) else 0)

to_number(power,[['power'],['efficiency',None],['vent',0],['sata',0],
                 ['molex',0],['pci_e',0],['+3.3V',None], ['+5V',None], 
                 ['+12V1',None], ['-12V',None], ['+5Vsb',None], 
                 ['+12V_watt',None],['+3.3V_+5V_watt',None], ['warranty',None],
                 
                 ])
power.cool = power.cool.apply(lambda x: 1 if '1' in x else(2 if '2' in x else 0))

power.sertificate = power.sertificate.apply(
                     lambda x:x.split(' /')[0] if x == x else 'без 80+')
extract_dimensions(power)
power.to_csv('data/datasets/power.csv',index = False)


hdd = pd.read_csv('data/raw_data/hdd.csv')
hdd = hdd[hdd['Тип накопителя'] == 'HDD']
hdd = hdd[['id', 'name', 'price', 'link', 'Исполнение', 'Назначение', 'Объем',
       'Форм-фактор', 'Интерфейсы подключения', 'Объем буфера обмена',
       'Частота вращения шпинделя', 'Среднее время поиска',
       'Ударостойкость при работе', 'Уровень шума при чтении',
       'Уровень шума в режиме ожидания', 'Размеры', 'Вес',
       'Дата добавления на E-Katalog','Гарантия производителя',
       'Потребляемая мощность при работе','Потребляемая мощность при ожидании']]
hdd.columns = ['id', 'name', 'price', 'link', 'inner','purpose', 'volume',
       'form','interface',  'buffer','spindel', 'search_time','resistance',
       'noise_read','noise_wait', 'dimensions', 'weight', 'age',
       'warranty', 'consume_work','consume_wait']

hdd.dropna(subset = ['inner','purpose','volume','form','interface'],inplace = True)
age(hdd)
hdd.inner = hdd.inner.apply(lambda x: 1 if 'внеш' in x else 0)
hdd.spindel = hdd.spindel.apply(
    lambda x: '7200 ' if x == x and (('7200' in x) or ('измен' in x)) else x)

to_number(hdd, [['volume'],['form'],['buffer',None],['spindel',None],
                ['search_time',None],['resistance',None],['noise_read',None],
                ['noise_wait',None],['weight',None],['warranty',None],
                ['consume_work',None], ['consume_wait',None]
                ])

for i in ['IDE','SAS','SATA','SATA 2','SATA 3','USB 3.2 gen1','USB 3.2 gen2',
'USB C 3.2 gen1','USB C 3.2 gen2','Thunderbolt v3','USB 2.0',  
'micro-USB','IEEE 1394','Fibre Channel']:
  hdd[i] = hdd.interface.apply(lambda x: 1 if i in x else 0)

hdd.drop('interface',axis=1,inplace = True)
extract_dimensions(hdd)
hdd.to_csv('data/datasets/hdd.csv',index = False)

cool = pd.read_csv('data/raw_data/cool.csv')
cool = cool[['id', 'name', 'price', 'link', 'Назначение', 'Тип', 
       'Вентиляторов', 'Тип крепления', 'Диаметр вентилятора',
       'Тип подшипника', 'Минимальные обороты', 'Максимальные обороты',
       'Регулятор оборотов', 'Макс. воздушный поток', 'Выдув воздушного потока',
       'Наработка на отказ', 'Питание', 'Уровень шума', 'Габариты', 
       'Дата добавления на E-Katalog', 'Материал радиатора', 'Socket', 
       'Цвет подсветки','Тепловых трубок'
       ]]

cool.columns = ['id', 'name', 'price', 'link', 'purpose', 'type', 
       'n_vents', 'tier', 'diameter','bearing', 'min_turn', 'max_turn',
       'reg_turn', 'max_airflow','flow_direction','mtlb', 'power','noise', 'dimensions', 
       'age','material_rad', 'socket', 'lightening','pipes']

cool = cool[(cool.purpose == 'корпус') | (cool.purpose =='процессор')]
cool = cool[(cool['type'] == 'вентилятор') | (cool['type'] =='активный кулер')]
cool.dropna(subset = ['diameter','bearing','max_turn'],inplace=True)
age(cool)
extract_dimensions(cool)

to_number(cool,[['n_vents',1],['min_turn',0],['max_turn'],['diameter'],
                ['max_airflow',None],['mtlb',None],['noise',None],
                ['pipes',0]              
                ])

for c in ['bearing','tier','reg_turn','power','material_rad']:
 cool[c] = cool[c].apply(lambda x: x.split()[0] if x==x else x)   

cool.min_turn[cool.min_turn == 0] = \
cool.min_turn[cool.min_turn == 0].combine(cool.max_turn[cool.min_turn == 0],max)

cool.to_csv('data/datasets/cool.csv',index = False)

case = pd.read_csv('data/raw_data/case.csv')
for text in ['окно на боковой панели','скрытая протяжка проводов',
             'окно установки СО для процессора','закаленное стекло']:
  case[text] =case['Дополнительно*'].apply(lambda x: 1 if x== x and text in x else 0)

case = case[['id', 'name', 'price', 'link','Дата добавления на E-Katalog','Форм-фактор',
             'Тип материнской платы', 'Форм-фактор БП', 'Габариты (ВхШхГ)','Длина видеокарты, до', 
             'Высота кулера, до','Вес','Материал', 'Толщина боковых стенок','Расположение БП', 
             'Отсеков 5,25"', 'Внутренних отсеков 3,5"','Внешних отсеков 3,5"','Внутренних отсеков 2,5"',
             'Отверстий под слоты расширения','Мест для вентиляторов сзади', 'Мест для вентиляторов сбоку',
             'Мест для вентиляторов спереди','Мест для вентиляторов сверху','Мест для вентиляторов снизу',
             'Мест для вентиляторов','Установленных вентиляторов','Расположение', 'USB 2.0','USB 3.2 gen1',
             'USB 3.2 gen2','USB C 3.2 gen1','USB C 3.2 gen2','По направлению', 'Тип подсветки',
             'Мест для СВО','Мощность комплектного БП','окно на боковой панели','скрытая протяжка проводов',
             'окно установки СО для процессора','закаленное стекло']]

case.columns = ['id', 'name', 'price', 'link','age', 'form','mb_type', 'power_form',
       'dimensions','v_card_length', 'cooler_height', 'weight','mateial',
       'sidewall_thickness','PS_location', '5,25"_cell', '3,5"_cell','outer_3,5"',
       '2,5"_cell','expansion_slots','vents_back', 'vents_side','vents_front',
       'vents_up', 'vents_down', 'all_vents_places','mounted_vents','location',
       'usb_2','usb_3_gen1','usb_3_gen2','usb_c_gen1','usb_c_gen2','gaming', 
       'light', 'water','power','side_window','hidden_wire',
       'cool_window','strained_glass']

case.dropna(subset = ['mb_type','form','power_form',
                      'PS_location'],inplace=True)
age(case)

for c in ['form','mateial','location']:
  case[c] = case[c].apply(lambda x:x.split(' /')[0] if x == x else x)
extract_dimensions(case)
to_number(case,[['v_card_length',None],['cooler_height',None],['weight',None],
                ['sidewall_thickness',None],['5,25"_cell',0], ['3,5"_cell',0],
                ['2,5"_cell',0],['expansion_slots',0],['all_vents_places',0],   
                ['outer_3,5"',0],['mounted_vents',0],['usb_2',0],
                ['usb_3_gen1',0],['usb_3_gen2',0],['usb_c_gen1',0],
                ['usb_c_gen2',0],['water',0],['power',0]
                ])

for c in ['vents_back', 'vents_side','vents_front',
       'vents_up', 'vents_down']:
 case[c] = case[c].str.findall(r'\d\D\d+').apply(full_diameter)
case['usb_3'] = case.usb_3_gen1 + case.usb_3_gen2
case['usb_c'] = case.usb_c_gen1 + case.usb_c_gen2

case.drop(['usb_3_gen1','usb_3_gen2',
           'usb_c_gen1','usb_c_gen2'],axis = 1,inplace = True)
to_bin(case,['gaming','light'])

case.to_csv('data/datasets/case.csv',index = False)


note = pd.read_csv('data/raw_data/notebooks.csv')
note.dropna(subset=['Частота смены кадров'],inplace=True)
note = note[['id', 'name','price', 'link', 'Тип', 'Диагональ экрана', 'Тип матрицы', 'Покрытие экрана',
 'Разрешение дисплея', 'Частота смены кадров', 'Яркость','Серия',  'Кодовое название',
 'Кол-во ядер', 'Кол-во потоков', 'Тактовая частота', 'Частота TurboBoost / TurboCore',
 'Объем кэш памяти 2-го уровня', 'Объем кэш памяти 3-го уровня', 'Тест Passmark CPU Mark',
 'Объем оперативной памяти', 'Максимально устанавливаемый объем', 'Тип памяти', 'Частота памяти',
 'Кол-во слотов', 'Тип видеокарты', 'Серия видеокарты', 'Модель видеокарты', 'Объем видеопамяти',
 'Тип памяти*', 'Тип накопителя', 'Емкость накопителя', 'Интерфейс накопителя M.2',
 'USB 2.0', 'USB 3.2 gen1','USB 3.2 gen2', 'USB C 3.2 gen1', 'USB C 3.2 gen2',
 'Wi-Fi', 'Web-камера', 'Количество динамиков',
  'Емкость батареи','Емкость батареи*', 'Макс. время работы', 
 'Предустановленная ОС',  'Габариты (ШхГхТ)', 'Вес','Дата добавления на E-Katalog', 'Тест 3DMark06',
 'Картридер', 'Дополнительный разъем M.2', 'USB4', 'Интерфейс Thunderbolt', 'Макс. подключаемых мониторов'
# 'Материал корпуса','Bluetooth',
# 'Быстрая зарядка','Кол-во ячеек батареи','Размер накопителя M.2', #'Брендовая акустика', 'Аудиодекодеры','Манипулятор','Напряжение батареи',
# 'Интерфейс доп. разъема M.2', 'Размер доп. накопителя M.2','Стандарт защиты MIL-STD-810', 'Цветовой охват (NTSC)', 'Цветовой охват (DCI P3)',
#  'Цветовой охват (Adobe RGB)', 'Дополнительных клавиш', 'Время отклика',#  'Стекло Gorilla Glass', 'LAN (RJ-45)','Время зарядки',
# 'Синхронизация подсветки', 'Интерфейс разъема M.2',#  'Комплектация', 'USB C 3.2 gen2x2', 'Вес (планшет)', 'Взаимодействие со смартфоном',
#  'Шторка для камеры', 'NFC-чип', 'Подключение док-станции', 'Сенсорный', 'Влагозащита','Модель','Порты подключения',  'Безопасность',
]]
note.columns = ['id', 'name','price', 'link', 'type', 'diagonal', 'matrix', 'screen_coverage',
 'display_res', 'frame_rate', 'bright','series', 'code_name', 'cores', 'threads', 'clock_freq', 'turbo_freq',
 'l2_cashe', 'l3_cashe', 'passmark_test','ram_vol', 'max_ram_vol', 'ddr4', 'ram_freq','ram_slots', 
 'discrete_video', 'video_series', 'video_model', 'video_ram_vol', 'video_ram_type', 'drive_type', 'drive_vol', 'm2_interface',
 'usb_2', 'usb_3_g1','usb_3_g2', 'usb_c_g1', 'usb_c_g2', 'wi_fi', 'camera', 'dynamics', 
 'bat_capacity','bat_capacity*', 'max_work_time','op_sys', 'dimensions', 'weight',
 'age', '3dmark','cardreader','add_m2', 'usb4', 'thunderbolt', 'max_monitors']

extract_dimensions(note)
age(note)
note.turbo_freq.fillna(note.clock_freq,inplace=True)
note.max_monitors.fillna(0,inplace=True)
note.max_ram_vol.fillna(note.ram_vol,inplace=True)
note.bat_capacity.fillna(note['bat_capacity*'],inplace=True)
note.drop('bat_capacity*',axis = 1,inplace = True)
note.dropna(subset = ['diagonal','matrix','screen_coverage','frame_rate','display_res',
                      'cores','clock_freq','ram_vol','ddr4','ram_freq','bat_capacity'],inplace=True)

note.threads.fillna(note.threads.mean(),inplace=True)
note.ram_slots.replace('встроенная + 1 слот',1,inplace=True)
note.ram_slots.replace('встроенная',0,inplace=True)
note.ram_slots = pd.to_numeric(note.ram_slots)
smart_fillna(note,'max_work_time','bat_capacity')
note.thunderbolt = note.thunderbolt.apply(lambda x: x.split(' ')[1] if x==x else x)


to_number(note,[['diagonal',None],['frame_rate',None],['bright',int(note.bright.mode()[0].split('\xa0')[0])],
                ['clock_freq',None],['turbo_freq',None],['l2_cashe',0],['l3_cashe',0],['ram_vol',None],
                ['passmark_test',int(note.passmark_test.mode()[0].split('\xa0')[0])],['max_ram_vol',None],
                ['ram_freq',None],['video_ram_vol',0],['drive_vol',0],['usb_2',0],['usb_3_g1',0],['usb_3_g2',0],
                ['usb_c_g1',0],['usb_c_g2',0],['dynamics',0],['bat_capacity',0],['weight',0],['add_m2',0],
                ['max_work_time',int(note.max_work_time.mode()[0].split('\xa0')[0])],['usb4',0],['thunderbolt',0],
                ['3dmark',int(note['3dmark'].mode()[0].split('\xa0')[0])]
                ])

note.bat_capacity = note.bat_capacity.apply(lambda x: x/1000*12 if x>1000 else x)  

note.ddr4 = note.ddr4.apply(lambda x:1 if 'DDR4' in x else 0)
note.discrete_video = note.discrete_video.apply(lambda x:1 if 'дискретная' in x else 0)
note.screen_coverage = note.screen_coverage.apply(lambda x: x.split(' /')[0])
note.display_res = note.display_res.apply(lambda x: int(x.split('x')[0]))
note.video_model = note.video_model.apply(lambda x: x.split(' ')[0])
note.m2_interface = note.m2_interface.apply(lambda x: x.split(' /')[0] if x==x else x)
note.camera = note.camera.apply(lambda x: 0 if 'отсутствует' in x else int(x.split('x')[0]))

cat_cols = ['matrix','screen_coverage','type','series','code_name','video_series',
            'video_model','video_ram_type','drive_type','m2_interface','wi_fi','op_sys',
            'cardreader']    
for c in cat_cols:
    encoder = ce.PolynomialEncoder()
    X= encoder.fit_transform(note[c],note.price)
    note.drop(c,axis = 1,inplace = True)
    note = note.join(X.drop('intercept',axis=1))
note.to_csv('data/datasets/notebooks.csv',index = False)

