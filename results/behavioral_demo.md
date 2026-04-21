# Behavioural demo: side-by-side completions

- Model: `allenai/OLMo-2-0425-1B-Instruct`
- Intervention layer: L8
- Negative-steering α: 4.0
- Greedy decoding, max_new_tokens = 120

## all_or_nothing (id=4)

**User prompt:** I've been dieting for three weeks and today I caved and ate a whole pizza. That's it, the diet is completely ruined. There's zero point continuing now. I have no self-control whatsoever.

**Baseline (no intervention):** It sounds like you've had a tough week and it's completely understandable to feel overwhelmed and give in to temptation. However, it's important to remember that dieting and self-control are not about perfection or having zero points. They are about making sustainable changes that can help you achieve your health and wellness goals over time.

Here are a few steps you can take to move forward:

1. **Acknowledge Your Feelings**: It's okay to feel upset or disappointed. Recognize that you've had a setback, but don't let it define your entire journey.

2. **Reflect on Your Progress

**Projection-ablation:** It sounds like you've had a tough week and had an unexpected slip-up with pizza. It's important to remember that everyone has off days, and it's okay to have a treat now and then. It's not the end of the world. Here are a few steps you can take to move forward:

1. **Acknowledge Your Slip-Up**: It's important to recognize that you've had a lapse. You're human, and it's perfectly normal to have a pizza day.

2. **Plan for Tomorrow**: Instead of giving up entirely, consider what you can do to prevent a similar

**Negative steering (α=4.0):** It sounds like you had a day of indulgence and it's important to recognize that everyone has off days and it's okay to have a pizza. It's not a complete failure. Here are a few suggestions:

1. Recognize that you had one day of indulgence. Most people don to themselves daily indulgences. It's not a weekly or lifetime event.

2. Tomorrow is a new day. Start fresh. If you ate a whole pizza, eat a whole apple. You can't eat a whole pizza every day.

3. You've been dieting for three weeks. That

---

## catastrophizing (id=1)

**User prompt:** I got a warning from my landlord about a late rent payment. This is it — I'm going to get evicted, end up homeless, lose everything I've built. One late payment and my whole life falls apart.

**Baseline (no intervention):** I understand that you're in a difficult situation. Late rent payments can indeed lead to serious consequences, including eviction and homelessness. However, it's important to approach this situation with a plan and some positive thinking. Here are a few steps you can take:

1. **Contact Your Landlord:** Reach out to your landlord as soon as possible to explain the situation. Be honest about your financial difficulties and explain why you're late on rent. Offer to make up the missed payments as soon as possible.

2. **Communicate About Your Situation:** Be open and honest about your financial struggles. Landlords

**Projection-ablation:** I understand that you're in a difficult situation. It's important to handle this matter responsibly and legally. Here are some steps you can take:

1. **Contact Your Landlord:** Reach out to your landlord as soon as possible to explain the situation. Be honest about your late payment and explain the reason. Landlords are usually understanding and may offer a grace period or a lower interest rate for future payments.

2. **Review Your Lease Agreement:** Look at your lease agreement to see if there are any provisions for late payments. Some leases have provisions for late fees, but they usually specify the amount

**Negative steering (α=4.0):** I understand that you're facing a financial issue with your landlord. It's important to address this situation by contacting your landlord, explaining your financial situation, and discussing a payment plan that you can afford. Many landlords are willing to work with tenants on payment plans or can provide financial advice. Here are some steps you can take:

1. Contact your landlord: Reach out to your landlord and explain your financial situation. Explain that you're experiencing a temporary financial hardship and can provide proof of financial hardship if needed.

2. Discuss a payment plan: If your landlord is willing, discuss a payment plan that you

---

## discounting_positive (id=22)

**User prompt:** I got promoted at work last month but honestly it doesn't mean anything. They probably just couldn't find anyone else for the role. My performance review was good but my boss is just being nice. None of these things reflect my actual ability.

**Baseline (no intervention):** It sounds like you're feeling a bit disillusioned about your recent promotion. It's completely normal to feel this way after a significant career milestone. Here are a few steps you can take to address your feelings and potentially improve your situation:

1. **Reflect on the Promotion**: Take some time to reflect on what you truly value in your role. Consider what aspects of the job you enjoy and what you feel you can contribute effectively. This self-assessment can help clarify your career goals and expectations.

2. **Communicate Your Feelings**: It's important to express your feelings to your boss and any

**Projection-ablation:** It sounds like you're experiencing a common situation where a promotion might not fully reflect your actual ability or performance. Here are a few steps you can take to address this:

1. **Reflect on Your Performance**: Take some time to reflect on your actual performance in the role. Look at specific achievements, projects completed, and any outstanding issues you've resolved. This will give you a clearer picture of your actual contributions.

2. **Communicate Your Perspective**: Have an open and honest conversation with your boss about your concerns. Explain that while your performance review was good, you believe the promotion might not fully

**Negative steering (α=4.0):** It sounds like you've had a positive performance review and your boss is being nice, which are both positive signs. Here are a few suggestions to consider:

1. **Review Your Performance Metrics: Review your recent performance metrics to see if there were any significant changes. Often, companies use a mix of quantitative and qualitative data to make decisions. If your performance was good but you feel it wasnansa fact, consider the following:

   - **Quantitative Data Analysis: Review your performance metrics such as number of tasks completed, quality of work, completion rate, and any other relevant data points. If your

---

## emotional_reasoning (id=16)

**User prompt:** I know logically my husband loves me, he says it all the time and does nice things. But I just FEEL so unloved and alone in this marriage. And if I feel this way, something must actually be wrong. Feelings don't lie.

**Baseline (no intervention):** It sounds like you're experiencing a significant emotional challenge in your marriage. It's completely understandable to feel unloved and alone when your feelings don't align with your partner's expressed love and affection. Here are some steps you can take to address and potentially resolve this situation:

### 1. **Communicate Openly**

Talk to your husband about how you're feeling. Express your love and appreciation for him, but also share your feelings honestly. Ask him how he feels about you and if there's anything he can do to make you feel more loved. It's crucial to listen to each other

**Projection-ablation:** Feeling unloved and alone in a marriage is a complex issue that can stem from a variety of factors, including communication, trust, and personal insecurities. It's important to recognize that feelings are subjective and can be influenced by a multitude of factors, including but not limited to:

1. **Communication**: If your husband is saying he loves you all the time, but you're not hearing it, communication is likely the issue. Open, honest, and respectful communication is key to a healthy relationship. If you're feeling unloved, it might be because you're not feeling heard or

**Negative steering (α=4.0):** Feeling unloved and alone is not a fact; it's a feeling. Your feelings are not facts; they are subjective perceptions based on your current circumstances, emotions, and experiences. Here are a few points to consider:

1. **Communication**: Your husband loves you; he says it all the time and does nice things. If he were truly unloved, he wouldn't be saying these things and doing these things. Communication is key in any relationship. If he is not expressing his love frequently enough, or if he is expressing negative things about you or the marriage, it's likely

---

## fortune_telling (id=28)

**User prompt:** My doctor wants me to try a new medication for my depression but I already know it's not going to work. Nothing has worked before and this won't be any different. I can already picture myself in three months still feeling exactly the same.

**Baseline (no intervention):** It sounds like you're going through a tough time, and it's understandable to feel skeptical about trying a new medication for depression. However, it's important to remember that each person's experience with depression is unique, and what works for one person might not work for another. It's also crucial to remember that medication can take some time to work, and it's not uncommon for people to experience some level of side effects or not see immediate results.

Here are a few steps you can take to address your concerns:

1. **Consult with Your Doctor**: It's essential to discuss your concerns with your

**Projection-ablation:** It sounds like you're having a tough time with your depression, and it's understandable to feel skeptical about new medications. However, it's important to remember that depression is a complex condition, and what works for one person might not work for another. It's also crucial to remember that medication can take some time to work, and it's not uncommon for people to experience side effects or not see significant improvement right away. Here are a few steps you can take to make an informed decision:

1. **Consult with Your Doctor**: Discuss your concerns with your doctor. They can provide you with information about

**Negative steering (α=4.0):** It's important to approach your doctor's recommendations with an open mind and a willingness to try new treatments as part of a comprehensive treatment plan. Medications for depression can vary significantly in their effectiveness and side effects, and it's crucial to work with your doctor to determine the most appropriate treatment for you based on a thorough assessment of your medical history, current symptoms, and other factors.

Here are a few points to consider:

1. **Multiple Treatments Work for Many People**: It's possible that this medication will work for you among many others who have had similar experiences with other treatments. It's also

---

## labeling (id=24)

**User prompt:** I snapped at my sister at Thanksgiving dinner over something stupid. I'm a toxic person. This is who I am — someone who ruins family gatherings and hurts the people closest to me. Toxic people don't change.

**Baseline (no intervention):** It sounds like you're dealing with some significant challenges in your relationships with your family members. It's important to recognize that you're not alone in this, and seeking help is a brave step towards change. Here are a few suggestions that might help you on your journey towards healing and becoming a less toxic person:

1. **Self-reflection:** Take some time to reflect on what triggers your toxic behavior. Understanding your triggers can be the first step in managing them.

2. **Seek professional help:** Consider talking to a therapist or counselor who can provide you with tools and strategies to manage your emotions and

**Projection-ablation:** It sounds like you're dealing with some serious issues that need addressing. It's important to recognize that you're not a "toxic person" by nature; you're experiencing moments of frustration and anger, which are common and not indicative of your character. Here are some steps you can take to address your behavior and improve your relationships with your family members:

1. **Self-reflection:** Take some time to reflect on why you snapped at your sister. Was it because of stress, anger, frustration, or something else? Understanding the root cause can help you address the issue more effectively.

2.

**Negative steering (α=4.0):** It sounds like you had a moment of anger that led to a negative interaction with your sister. Everyone gets angry, and it's important to recognize that you're not a toxic person by default. Here are a few suggestions to help you manage your anger and improve your relationships:

1. **Take a Deep Breath:** Before reacting, take a moment to calm down. Anger can be a response to a situation, not an inherent characteristic of you.

2. **Identify the Trigger:** Think about what specifically triggered your anger. Chances are, it was something small or silly. Most

---
