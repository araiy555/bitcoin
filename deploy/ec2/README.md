# EC2 で24時間録画し、毎朝検証して Slack に送る

サーバーが bitbank（と先行市場の Binance）を24時間録画し、毎朝9時20分（日本時間）に
前日1日分を固定設定で検証して、結果を Slack に送ります。`10秒内計` が2日続けて
マイナスになると、Slack に警告が出ます。

鍵はサーバーに置きません。S3 へのアクセスは EC2 に付ける IAM ロールで行うので、
`aws login` もログイン切れもありません。

## 1. IAM ロールを作る（S3 の jsboard-capture だけ使える）

1. AWS コンソール → IAM → ポリシー → ポリシーの作成 → JSON
2. このフォルダの `iam-policy.json` の中身を貼り付け、名前を `jsboard-s3` にして作成
3. IAM → ロール → ロールを作成 → 信頼されたエンティティ「AWS のサービス」→ ユースケース「EC2」
4. ポリシー `jsboard-s3` を付けて、名前を `jsboard-ec2` にして作成

## 2. EC2 を作る

AWS コンソール → EC2（リージョンは **東京 ap-northeast-1**）→ インスタンスを起動

| 項目 | 設定 |
|---|---|
| AMI | Amazon Linux 2023 |
| インスタンスタイプ | **t3.micro**（無料枠） |
| キーペア | 新規作成してダウンロード（SSH 用） |
| ネットワーク | SSH を「自分の IP」からだけ許可 |
| ストレージ | 16GB gp3 |
| 高度な詳細 → IAM インスタンスプロフィール | **jsboard-ec2** |
| 高度な詳細 → クレジット仕様 | **スタンダード**（無制限のままだと超過分が課金される） |

## 3. サーバーに入って準備する

```
ssh -i ダウンロードした鍵.pem ec2-user@サーバーのIP
sudo dnf install -y git
git clone -b claude/crypto-jane-street-board-bnf6m1 https://github.com/araiy555/bitcoin.git
bash bitcoin/deploy/ec2/setup.sh
```

リポジトリが非公開なら、`git clone` でユーザー名とトークンを聞かれます。GitHub の
Settings → Developer settings → Fine-grained tokens で、このリポジトリの読み取り
（Contents: Read-only）だけのトークンを作って使ってください。

## 4. Slack の URL をサーバーに入れる

```
sudo nano /etc/jsboard.env
```

`SLACK_WEBHOOK_URL=` の後ろに URL を貼って保存します（Ctrl+O → Enter → Ctrl+X）。
このファイルは root しか読めません。URL はチャットやリポジトリには書かないでください。

録画する銘柄は同じファイルの `JSBOARD_TARGETS` で変えられます。

## 5. 動かす

```
sudo systemctl enable --now jsboard-capture@bitbank:ada_jpy jsboard-capture@bitbank:sui_jpy
sudo systemctl enable --now jsboard-daily.timer
```

銘柄を増やすときは `jsboard-capture@bitbank:xlm_jpy` のように足して、
`/etc/jsboard.env` の `JSBOARD_TARGETS` にも同じ銘柄を足します。

## 紙上トレード（本物の板・仮想の注文）

毎朝の検証と同じ固定設定で、本物の板を見ながら仮想の注文を24時間出し続けます。
**注文は一切出しません。** 1日ごとの損益と約定の回数が、UTC の日付が変わった直後
（日本時間9時）に Slack に届きます。損失が3,000円に達するか、何かで止まった場合も
Slack に知らせます（止まっても30秒後に自動で再開します）。

```
sudo systemctl enable --now jsboard-paper@bitbank:ada_jpy
sudo journalctl -u 'jsboard-paper@*' -n 20 --no-pager    # 5分ごとの損益
```

t2.micro はメモリが 1GB しかないので、紙上トレードはまず1銘柄から始めてください。

## 確認

```
systemctl status 'jsboard-capture@*'           # 録画が動いているか
journalctl -u 'jsboard-capture@*' -n 20         # 録画のログ
systemctl list-timers jsboard-daily.timer       # 次の検証の時刻
sudo systemctl start jsboard-daily              # 検証を今すぐ1回動かす（前日分）
journalctl -u jsboard-daily -n 30               # 検証の結果
```

## コードを更新する

```
cd ~/bitcoin && git pull && bash deploy/ec2/setup.sh
sudo systemctl restart 'jsboard-capture@*'
```

## 費用の注意

- 無料枠が終わると課金が始まります。AWS の Billing → 予算 で、月1,000円などの予算アラートを作っておいてください。
- 録画データは S3 の `raw/` に入り、7日で自動的に消えます。日次の結果は `reports/daily/` に残ります。
