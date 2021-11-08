# Diamond-needle-quality 

### Автоматизированное определение степени износа иглы

## Описание проекта:

"Гравировальные станки с ЧПУ (числовым программным управлением) используют алмазные иглы (на конце закреплен алмаз) для выполнения гравировки. [Пример игл](https://image.jimcdn.com/app/cms/image/transf/none/path/sd06fad7c1705147a/image/ie12ec70ba97a61b5/version/1431781986/image.jpg). В процессе работы алмаз тупиться или могут образовываться различные дефекты: сколы, трещины и др. Периодически нужно контролировать износ алмаза и при необходимости затачивать его.

Контроль производиться с помощью электронно микроскопа с которого можно получить увеличенное изображение иглы. После чего износ иглы определяется "на глаз" или с использованием простейших программных средств, например, приложений, определяющих угол между прямыми, которые пользователь выстывит вручную. Стоит отметить, что это ПО зачастую ""заточено"" под конкретны станок.

Процесс определения степени износа иглы и рекомендации по её заточке можно автоматизировать, применив алгоритмы обработки изображений (например, выделение контуров). Кроме того универсальность данного ПО (возможность работы с изображениями с разных USB микроскопов) будет дополнительным плюсом.

[Пример затупленной и переточенной иглы.](https://image.jimcdn.com/app/cms/image/transf/none/path/sd06fad7c1705147a/image/ie12ec70ba97a61b5/version/1431781986/image.jpg)

## Ход работы:

Для решения поставленной задачи было решено использовать классические алгоритмы компьютерного зрения для получения контура иглы, по которому в последующем происходит построение прямых и оценка тупости иглы. Всвязи с первоначальным отсутствием датасета, в начале работы был собран датасет, состоящий из 13 картинок тупых и заточенных алмазных игл, найденных на просторе интернета. Для упрощения поставленной задачи, всвязи с малым объёмом собранного датасета, было решено избавиться от фона изображений.

Опишем проделанные шаги:

### 1. Применение фильтрации к изображению

### 2. Применение алгоритмов выделения границ

В качестве алгоритмов выделения границ были реализовны следующие алгоритмы:
* фильтр Кэнни;
* оператора Собеля;
* оператор Робертса;
* оператор Прюитт;

Из всех релаизованных алгоритмов самым лучшим был призван фильтр Кэнни. 
Пример его работы отражён в изображении ниже. 

![image](https://user-images.githubusercontent.com/54993653/140667661-57d7b048-a787-4ab9-a52d-ed2ab079b528.png)

### 3. Получение и отсеивание границ

После применение алгоритмов выделения границ из изображения выделенных границ вытаскивались контуры, после чего последние фильтровались по минимальной длине.

После этого полученные контуры объединялись, из них брались самая нижние точки и переводились в таблицу Pandas - DataFrame.

Пример получения контуров и их фильтрации отражён в изображении ниже.
![image](https://user-images.githubusercontent.com/54993653/140668007-d766c4bc-5058-4e34-a1b0-14076a604208.png)

### 4. Построение прямых, описывающих границы

После получения контура иглы на контуре находила данные контур разбивался на левую и правую часть, находилось максимальная точка контура (которая, по предположению, должна быть нужным концом иглы). Контуры левой и правой частей прогонялись через sklearn LinearRegression, тем самым мы получали функцию линейной интерполяции. После этого веса данных моделей вытаскивались из них и строились следующие 3 прямые:
* Левая граница: $y=w1x+w2$,
* Правая граница: $y=w1x+w2$,
* Горизонтальная граница: $y=y_max$,
* 
где \\( w1, w2 )\\ - соответствующие линейным моделям веса, 
\\( y_max )\\  - найденная максимальная точка.

После нахождения уравнений 3х прямых, происходит вывод их на экран (жёлтыми точками показан принятый контур):

![image](https://user-images.githubusercontent.com/54993653/140669697-3631c142-c65c-4c73-9472-ff7b5f3cfe5e.png)
