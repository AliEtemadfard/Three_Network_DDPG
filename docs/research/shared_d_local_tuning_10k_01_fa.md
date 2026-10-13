# برنامه‌ی جست‌وجوی محلی پارامترها ــ نسخه‌ی 10k-01

## نتیجه‌ی noise=0.3

افزایش noise از `0.1` به `0.3` نتیجه‌ی نهایی را از حدود `-1115.8` به
`-996.5` رساند و علامت مشتق Q2 نسبت به action را روی grid ثابت از حالت
یک‌طرفه به تقریباً متوازن تغییر داد. بااین‌حال AUC از `-1358.1` به `-1360.7`
رسید و بهتر نشد. بنابراین اثر مثبت دیرهنگام است و هنوز نامزد انتقال به محیط
دوم نداریم.

## screen محدود

سه اجرای تک-seed انجام می‌شود:

| variant | noise std | Actor LR | سؤال |
| --- | ---: | ---: | --- |
| `noise-0p2-actor-3e-5` | 0.2 | `3e-5` | آیا پوشش میانی زودتر و پایدارتر یاد می‌گیرد؟ |
| `noise-0p4-actor-3e-5` | 0.4 | `3e-5` | آیا پوشش بیشتر action مفید است یا فقط variance را زیاد می‌کند؟ |
| `noise-0p3-actor-5e-5` | 0.3 | `5e-5` | آیا Actor کمی سریع‌تر از action-gradient بهترشده استفاده می‌کند؟ |

تمام اجراها seed آموزشی `11`، بودجه‌ی ۱۰٬۰۰۰ قدم، Q1/Q2 LR=`1e-4`،
delay=`10`، `alpha=0.01` و `alpha_prime=10` دارند. seedهای ارزیابی
`1001–1005` و تمام فرمول‌ها، minibatchها و ترتیب update ثابت‌اند.

این مرحله ۳ run و مجموعاً ۳۰٬۰۰۰ قدم دارد. نتیجه فقط «بهترین تنظیم در این
screen محلی و seed 11» خواهد بود؛ نه بهترین پارامتر عمومی.

## معیار انتخاب

رتبه‌بندی از قبل ثابت است:

1. AUC بهتر؛
2. میانگین پنج checkpoint آخر بهتر؛
3. نبود collapse و متریک‌های مکانیزمی آرام‌تر؛
4. پروفایل action-gradient نهایی بدون علامت غالب غیرمنطقی.

پس از این سه run نقطه‌ی دیگری به grid اضافه نمی‌شود. اگر هیچ نسخه‌ای بهبود
واضح و پایدار نداشت، sweep پارامتر متوقف و خود سیگنال Q2/Actor بازبینی می‌شود.

## اجرای دستی PowerShell

```powershell
Set-Location 'F:\lessons\arshad\three-network-ddpg'

$Python = 'C:\Users\aliet\anaconda3\envs\three-network-ddpg\python.exe'
$Manifest = 'experiments\manifests\pendulum_shared_d_local_tuning_10k_01.yaml'

& $Python -m three_network_ddpg experiment validate $Manifest
& $Python -m three_network_ddpg experiment plan $Manifest
& $Python -m three_network_ddpg experiment run $Manifest
```

پس از پایان:

```powershell
& $Python -m three_network_ddpg experiment status $Manifest
& $Python -m three_network_ddpg experiment aggregate $Manifest

& $Python -m three_network_ddpg experiment plot $Manifest --metric evaluation_return_mean
& $Python -m three_network_ddpg experiment plot $Manifest --metric shared_squared_error
& $Python -m three_network_ddpg experiment plot $Manifest --metric q_abs_max
& $Python -m three_network_ddpg experiment plot $Manifest --metric actor_gradient_norm
& $Python -m three_network_ddpg experiment plot $Manifest --metric critic_one_gradient_norm
& $Python -m three_network_ddpg experiment plot $Manifest --metric critic_two_gradient_norm
& $Python -m three_network_ddpg experiment plot $Manifest --metric actor_next_action_utilization_mean
```

در صورت interruption واقعی:

```powershell
& $Python -m three_network_ddpg experiment resume $Manifest
```

## نتیجه‌ی اجرا ــ ۲۰۲۶-۱۰-۱۳

هر سه run با seed آموزشی `11` کامل و finite شدند؛ هیچ run ناموفق، ناقص،
divergent یا non-finite نبود. هر run شامل ۱۰٬۰۰۰ قدم محیط، ۹٬۰۰۰ update و
۲۰ checkpoint ارزیابی پنج‌اپیزودی بود.

| تنظیم | normalized AUC | بهترین return (قدم) | return نهایی | میانگین ۵ ارزیابی آخر |
| --- | ---: | ---: | ---: | ---: |
| baseline: noise=`0.3`, Actor LR=`3e-5` | `-1360.7` | `-939.3` (9500) | **`-996.5`** | **`-1078.9`** |
| noise=`0.2`, Actor LR=`3e-5` | `-1360.9` | `-994.3` (9000) | `-1049.2` | `-1095.7` |
| noise=`0.3`, Actor LR=`5e-5` | **`-1356.7`** | `-1029.0` (9500) | `-1068.4` | `-1094.8` |
| noise=`0.4`, Actor LR=`3e-5` | `-1375.8` | **`-873.1`** (9500) | `-1032.1` | `-1125.2` |

بهترین AUC متعلق به Actor LR=`5e-5` است، ولی اختلاف آن با baseline فقط حدود
چهار واحد return است و در همان run مقدار نهایی، میانگین پنج ارزیابی آخر و بهترین
checkpoint همگی بدترند. بنابراین این تفاوت تک-seed برای انتخاب نرخ یادگیری جدید
کافی نیست. noise=`0.4` نیز در قدم ۹۵۰۰ یک جهش خوب داشت، اما در checkpoint بعدی
حدود ۱۵۹ واحد افت کرد و AUC و میانگین انتهایی بدتری داشت؛ این رفتار شواهد برتری
پایدار نیست.

میانگین diagnostics در ۵۰۰ update آخر:

| تنظیم | shared error² | Q2 TD error² | agreement current / next | mean abs(Q) | grad Q1 | grad Q2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `0.3 / 3e-5` | `6.44` | `19.04` | `6.47 / 6.35` | `439.18` | `216.53` | `2140.02` |
| `0.2 / 3e-5` | **`5.66`** | **`15.45`** | **`5.01 / 5.64`** | `422.53` | `191.32` | `1788.65` |
| `0.3 / 5e-5` | `7.35` | `19.82` | `5.87 / 6.54` | **`415.11`** | **`184.33`** | **`1716.03`** |
| `0.4 / 3e-5` | `7.03` | `21.05` | `7.70 / 7.79` | `420.11` | `223.28` | `2280.60` |

noise=`0.2` خطاها و norm گرادیان criticها را کاهش داد، اما return را بهتر نکرد.
در هر چهار تنظیم، norm گرادیان Q2 تقریباً ۹ تا ۱۰ برابر Q1 باقی ماند. پس تغییر
noise یا Actor LR عدم‌تعادل اصلی criticها را برطرف نکرده است.

### تصمیم

در این screen محلی «برنده‌ی قطعی» وجود ندارد. برای ادامه‌ی کار، baseline
noise=`0.3` و Actor LR=`3e-5` متعادل‌ترین نامزد فعلی است، چون بهترین رفتار
انتهایی را دارد. Actor LR=`5e-5` فقط نامزد AUC است و قبل از استفاده باید روی
seedهای مستقل تأیید شود. فعلاً گسترش sweep نویز/Actor LR توصیه نمی‌شود؛ نسبت
گرادیان Q2/Q1 و مقیاس بزرگ Qها نشان می‌دهد محدودیت اصلی احتمالاً در دینامیک
criticها است، نه صرفاً exploration یا سرعت Actor.
