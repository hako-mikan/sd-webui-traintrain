# TrainTrain
- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)用の拡張です
- LoRA、iLECO及び差分LoRAを作成できます

[<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](README.md)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](#overview)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)

# Overview
Stable DiffusionのLoRAを学習するツールです。Stable Diffusion Web-UIの拡張として動作し、学習用の環境構築を必要としません。通常のLoRA及び、モデルの概念を除去・強調するLECOの学習を高速化したiLECO(instant-LECO)と、ふたつの差分画像からスライダーLoRAなどを作成する差分学習を行えます。

# Recent Update
2025.03.04
新しい学習モード「ADDifT」「Multi-ADDifT」を追加しました。詳細は[Noteの記事]()を参照してください。

## もくじ
- [使用要件](#使用要件)
- [インストール](#インストール)
- [使い方](#使い方)
    - [LoRA](#lora)
    - [iLECO](#ileco)
    - [Difference](#difference)
    - [ADDifT](#addift)
- [設定](#設定)
    - [必須パラメーター](#必須パラメーター)
    - [オプションパラメーター](#オプションパラメーター)
- [Queue(学習の予約)](#queue)
- [Plot](#plot)
- [謝辞・参考文献](#謝辞)

## 使用要件
　Web-UI 1.7以上、最新版のForgeで動作します。

## インストール
　Web-UIのInstall From URLに`https://github.com/hako-mikan/sd-webui-traintrain`と入力しInstallボタンを押します。少し(数秒～数十秒)時間が掛かります。

## 使い方
　モードごとの必須パラメーターを入力しStart Trainingボタンを押すと学習が始まります。作成されたLoRAはLoRA用のフォルダに保存されます。モデルとVAEを選択しない場合、現在ロードされているモデルとVAEが使われます。
## LoRA
画像からLoRAを学習します。
### 入力画像
　`jpg`, `jpeg`, `png`, `gif`, `tif`, `tiff`, `bmp`, `webp`, `pcx`, `ico`形式に対応します。大きさは`image size`で指定した大きさにする必要はありませんが、学習時に切り取られるため切り取られ方によってはキャプションと不整合が生じる場合があるのである程度は整形した方がいいです。画像はアスペクト比ごとに分類されます。例えば`image size`を768,512に設定した場合、768×512のピクセルサイズを最大値としていくつかの解像度のセット（bucket）が作成されます。デフォルトの場合、768×512,512×512,512×768の3種類の分類になり、画像はアスペクト比ごとに近い分類に振り分けられます。これは学習が同一のサイズでないと受け付けないためです。そのさい、画像の縮小と切り取りが行われます。画像の中心を基準点として切り取りが行われます。分類を細分化したい場合には`image buckets step`の値を小さくします。

### 画像の縮小・ミラーリング
　同じ画像を何度も学習すると過学習になり、その画像そのものが出てきます。学習画像が少ない場合、過学習に対処するために画像を縮小したり反転させたりして学習画像を増やすことをします。`image size`を`768,512`に`image buckets step`を`128`に設定すると、
`(384, 768), (512, 768), (512, 640), (512, 512), (640, 512), (768, 512), (768, 384)`の枠が作られます。さらに、`image min length`を`256`にすると縮小用の枠として`(256, 512), (384, 640), (256, 384), (384, 512), (384, 384), (256, 256), (512, 384), (384, 256), (640, 384), (512, 256)`の枠が作られます。画像はまず普通の枠に振り分けられますが、`sub image num`が設定されている場合、アスペクト比の近い縮小用の枠にも縮小して格納されます。このとき、`(512, 640)`の枠に格納された画像について、`sub image num`が3と設定されている場合は `(384, 640)`、`(256, 384)`、`(384, 512)`にも縮小して格納されます。`image mirroring`を有効にすると左右反転された画像も格納され、結果として1枚の画像から8枚の学習用画像が生成されます。

### キャプション、トリガーワード
　画像と同じファイル名の`txt`,`caption`ファイルがある場合、ファイルに書かれたテキストを使って学習を行います。どちらも存在する場合、`txt`ファイルが優先されます。`trigger word`が設定されている場合、すべてのキャプションの前に`trigger word`が挿入されます。キャプションファイルがない場合も同様です。

### キャプションの考え方
　Aというキャラクターを学習するとします。Aというキャラクターはツインテールで青いシャツを着て赤いスカートを着用しているとします。Aが描かれた白い背景の絵があるとして、キャプションにはAの名前と、どの方向を向いているか、背景が白いことなどを記入します。ツインテールで青いシャツを着て赤いスカートを着用しているという要素はA特有の要素で学習させたいものなのでキャプションには記入しません。一方、向きや背景、構図などは学習してもらっては困るので記入します。

## iLECO
　iLECO(instant-LECO)はLECOの学習過程を高速化したもので、Original Promptで指定した概念をTarget Promptの概念に近づけるような学習を行います。Target Promptに何も入れない場合、その概念を除去するような学習になります。
　例としてどんなモデルでも強固に出てくるMona Lisaさんを消してみます。Original Promptに「Mona Lisa」、Target Promptは空欄にします。`train iteration` の値が500程度あれば収束します。`alpha`の値は通常rankより小さな値を設定しますが、iLECOの場合はrankより大きな値にした方がいい場合もあります。
 ![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample1.jpg)  
Mona Lisaさんを消すことができました。次にTarget Promptに「Vincent van Gogh Sunflowers」と入れてみます。すると、モナリザさんがひまわりになるLoRAができました。
![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample2.jpg) 
Original Promptに「red」、Target Promptに「blue」を入れてみます。赤を青くするLoRAができましたね。
 ![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample3.jpg) 
 
## Difference
　ふたつの差分画像からLoRAを作成します。いわゆるコピー機学習法というものです。いったん同じ画像しか出ないLoRA(コピー機)を作成した後、コピー機をLoRAを適用した状態で差分の学習を行うことで差分相当のLoRAをつくる方法です。Original, Targetに画像を設定してください。画像サイズは同じにしてください。
　まずコピー機の学習が始まり、その後差分の学習が始まります。例として目を閉じるLoRAを作ってみます。以下のふたつの画像を使います。  
   <img src="https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample4.jpg" width="200">
   <img src="https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample5.jpg" width="200">  
　Difference_Use2ndPassSettingsを使います。`train batch size`は1～3を設定します。大きな値を入れてもあまり意味はありません。できました。目を閉じる以外はほとんど画風や構図に影響を与えていません。これは2ndPassでrank(dim)を4と小さくしているためです。これをコピー機と同じ16にしてしまうと、画風や構図に影響を与えてしまいます。
 ![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample6.jpg) 

## ADDift
　ふたつの差分画像からLoRAを作成します。コピー機学習とは異なり、差分を直接LoRAに学習させるため高速に動作します。コピー機LoRAの学習は行いません。Original, Targetに画像を設定してください。画像サイズは同じにしてください。学習させたい対象によってmin/max timestepsを適切に設定しないとうまく学習が行われません。目を閉じる/開けるなどの動作や装飾などの場合にはMin=500, Max=1000にしてください。画風の場合にはMin = 200, Max = 400ぐらいがいいです。学習回数は30～100程度でよく、それ以上だと過学習になります。バッチサイズは1でいいです。バッチサイズを多くしても動作しますが、学習回数を少なくする必要があるのでバッチサイズを小さくして学習回数を稼いだ方が結果はいいと思います。

## Multi-ADDifT
　ふたつの画像の複数セットから差分LoRAを作成します。LoRA学習と同じようにディレクトリを指定します。ペアはファイル名で判断されます。ディレクトリ内で画像と「diff target name」で指定された画像のペアで学習を行います。例えば「diff target name」が「_closed_eyes」であるとき、「image1.png,image2.png」と「image1_closed_eyes.png,image2_closed_eyes.png」という画像のペアで学習がすすみます。LoRA学習と同じく読み込まれた画像はサイズごとにバケットに分配されます。詳細は[画像の縮小・ミラーリング](#画像の縮小・ミラーリング)を参照してください。

> [!TIP]
> VRAMが足りない場合は`gradient checkpointng`を有効化して下さい。計算時間が少し長くなる代わりにVRAM使用量を抑えられます。場合によっては`gradient checkpointng`を有効化してバッチサイズを大きくした方が計算時間が短くなる場合があります。コピー機学習ではバッチサイズを3より大きくしても変化は少ないので3以下の方がいいでしょう。バッチサイズは一度に学習する画像の数ですが、バッチサイズを倍にしたときに`iteration`を半分にできるかというとそう簡単な話ではありません。1ステップの学習で1回のウェイトの更新が行われますが、バッチサイズを倍にしてもこの回数は倍にはなりませんし、倍の効率で学習が行われるわけではないからです。

## 設定
## 必須パラメーター

|パラメーター| 詳細  | 
|----|----|
|network type | lierlaが普通のLoRAです。c3lier(いわゆるLoCON)やloha(いわゆるLyCORIS)だと学習する領域が増えます。c3lier, lohaを選んだ場合、オプションの`conv rank`と`conv alpha`を設定すると追加領域のdimを調節できます| 
|network rank | LoRAのサイズ。dimとも言います。大きすぎても良くないので16ぐらいから始めて下さい| 
|network alpha   |LoRAの縮小幅。通常rankと同じか小さな値を設定する。iLECOの場合はrankより大きな値を使った方がいい場合もあります| 
|lora data directory|LoRA学習を行う画像ファイルが保存されたフォルダを指定します。サブフォルダも含まれます| 
|lora trigger word|キャプションファイルを使用しないとき、ここに書かれたテキストと紐付けて学習が行われます。詳細は学習にて| 
|network blocks| 層別学習を行うときに使います。BASEはTextEncoderのことです。iLECO,DifferenceではBASEは使用しません| 
|train iterations  |学習回数。iLECO,Differenceだと500〜1000が適当です| 
|image size |学習時の解像度です。height, widthの順番はiLECOの時のみ有効になります| 
|train batch size |一度に何枚の画像を学習するかです。VRAMが共有メモリにはみ出ない程度に設定すると効率的です| 
|train learning rate|学習率です。iLECOだと1e-3〜1e-4、Differenceだと1e-3ぐらいが適当| 
|train optimizer|最適化関数の設定です。adamwが推奨。adamw8bitだと精度が落ちます。特にDifferenceではadamw8bitだとうまくいきません| 
|train lr scheduler|学習中に学習率を変化させる設定です。cosineにしとけばいいです。optimizerにadafactorを選ぶと学習率は自動的に決定されるためにこの項目は無効化されます| 
|save lora name |保存時のファイル名です。設定しないとuntitledになります| 
|use gradient checkpointng |VRAM使用量が抑えられる代わりに少し学習が遅くなる| 

## オプションパラメーター
オプションなので指定しなくても動作します。
|パラメーター| 詳細  | 
|----|----|
|network conv rank|c3lier, loha使用時のconv層のrank、0にするとnetwork rankの値が使われる| 
|network conv alpha|c3lier, loha使用時のconv層の縮小幅、0にするとnetwork alphaの値が使われる| 
|network element |学習対象を細かく指定します。lohaでは動作しません。<br>Full : 通常のLoRAと同じです<br>CrossAttention : プロンプトによって生成を処理する層のみを有効化します<br>SelfAttention : プロンプトを使わず生成を処理する層のみを有効化します|
|train min timesteps|学習を行う最小timesteps| 
|train max timestep|学習を行う最大timesteps| 
|train lr step rules|lr schedulerをstepにしたときのステップを指定| 
|train lr scheduler num cycles|cosine with restartの反復回数|
|train lr scheduler power |lrスケジューラーをlinearにしたときの指数|
|train lr warmup steps|学習初期に徐々にlrを上げていく時の有効ステップ数を指定| 
|train textencoder learning rate|Text Encoderの学習率、0だとtrain learning rateの値が使われる|
|image buckets step|画像をいくつかのアスペクト比に分類するときの分類の細かさを指定します|
|image min length|最小の解像度を指定します|
|image max ratio|最大のアスペクト比を指定します|
|sub image num |画像を異なる解像度に縮小する回数|
|image mirroring |画像を左右反転する|
|save per steps |指定ステップごとにLoRAを保存します| 
|save overwrite|保存時上書きするかどうか|
|save as json|学習実行時に設定を保存するかどうか。設定は拡張のフォルダのjsonフォルダに日付ごとに保存される|
|model v pred|SD2.Xモデルがv-predを使用するかどうか|
|train model precision|学習時の学習対象以外の精度。fp16で問題ない| 
|train lora precision|学習時の学習対象の精度。fp32で問題ない| 
|save precision|保存時の精度。fp16で問題ない| 
|train seed|学習時に使われるシード|
|diff save 1st pass|コピー機LoRAを保存するかどうか| 
|diff 1st pass only|コピー機LoRAのみを学習する| 
|diff load 1st pass|コピー機LoRAをファイルから読み込む| 
|train snr gamma|timestep補正を加えるかどうか。0~20の値を設定する。推奨値は5|
|logging verbose|コマンドプロンプトにログを出力する|
|logging_save_csv|csv形式でstep,loss,learning rateを記録する|

## プリセット、設定の保存とロード
　ボタンで設定を呼び出せます。設定はjsonファイルで扱います。プリセットはpresetフォルダに保存してあります。

## Queue
　学習を予約できます。`Add to Queue`ボタンを押すと、現在の設定で学習が予約されます。学習中にこのボタンを押すと、学習後に次の学習が自動的に始まります。学習前に押すと`Start Training`を押したときの設定で学習が終わった後、Queueリストの学習が順に処理されます。`save lora name`が同じ設定は追加できません。

## Plot
    logging_save_csvオプションを有効化したとき、学習の進捗をグラフ化できます。`Name of logfile`に何も入力しない場合、学習中か直近の学習の結果が表示されます。csvファイル名を入力するとその結果が表示されます。フルパスでは無くファイル名のみで大丈夫です。ファイルはlogsフォルダに入っている必要があります。

## 謝辞
　このコードは[Plat](https://github.com/p1atdev)氏の[LECO](https://github.com/p1atdev/LECO), [laksjdjf](https://github.com/laksjdjf)氏の[学習コード](https://github.com/laksjdjf/sd-trainer), [kohya](https://github.com/kohya-ss)氏の[学習コード](https://github.com/kohya-ss/sd-scripts)、[KohakuBlueleaf](https://github.com/KohakuBlueleaf)氏の[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)を参考にしています。

## Updates
2025.01.09
- Optimizerを追加しました。
DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, SGDNesterov8bit, Lion8bit, PagedAdamW8bit, PagedLion8bit, RAdamScheduleFree, AdamWScheduleFree, SGDScheduleFree, CAME, Tiger, AdamMini, PagedAdamW, PagedAdamW32bit, SGDNesterov
- Optimizer, lr Schedulerの追加設定を行えるようになりました。


## Reference
- https://github.com/rohitgandikota/erasing

- https://github.com/cloneofsimo/lora

- https://github.com/laksjdjf/sd-trainer

- https://github.com/kohya-ss/sd-scripts

- https://github.com/KohakuBlueleaf/LyCORIS

- https://github.com/ntc-ai/conceptmod
