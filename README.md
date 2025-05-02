# RAG Models Comparison
Chatting With Your Own Document - Building a Local RAG using LangChain and LangGraph with Various Open-Source LLM Models

<p align="center"><img src="https://github.com/user-attachments/assets/1ceb7ab2-e7cf-47dd-91ad-7d0e87a10c5c" width="500"/></p>

# Introduction

Imagine if Lionel Messi, Stephen Curry, Novak Djokovic, and a few other sports legends all showed up to play the same game. Chaos? Absolutely. Fairness? Questionable. Entertainment? Guaranteed. Now, swap those superstars for open-source Large Language Models (LLMs) and you’ve got yourself a showdown of epic AI proportions. Who will score the highest? Who will fumble? And who will just politely pass the ball? The curiosity is real, and so is the fun in seeing how these brainy bots behave when tasked with the same job.

# Comparison Objective

Retrieval-Augmented Generation (RAG) is a technique that makes AI language models, like ChatGPT to generate text with a smart search of relevant information from external sources. Without additional training, when we ask a question, our program will firstly search a special database of the document. The technique is called embedding which turns words and documents into numerical vectors. The retrieved information is then combined with our questions and fed into the language model which helps the AI to give answers. 

With different open-source LLM models in the market, it is interesting to see various relevance, accuracy and time-consumed of the responses by different models.

# Engaged Tools

<img src="https://github.com/user-attachments/assets/6fd5691a-63ad-44ac-bdd7-9bae3fc504aa" width="180"/>

<img src="https://github.com/user-attachments/assets/d1790845-a376-4fbe-bf1b-35364678b359" width="180"/>

<img src="https://github.com/user-attachments/assets/fc4a7c70-377c-4372-bff2-dc984a01058b" width="180"/>

<img src="https://github.com/user-attachments/assets/9b2a6f41-206b-4376-ba92-20ce3c8f3e3e" width="180"/>

<img src="https://github.com/user-attachments/assets/ebb4b5a4-a088-4839-9617-6a04ecb7a1b6" width="180"/>

# Key features

- Test run on a consumer-grade laptop (Windows 11 Home, AMD Ryzen 3.3GHz, 32GB Ram, NVIDIA GeForce RTX3050, 1TB HDD).
- No API key is needed.
- Document load from a text file from Internet.
- Text split and embed using LangChain HuggingFace.
- Pass the embedded text to vector dababase FAISS.
- Build the document retriever.
- Open-source LLMs connection by LangChain-Ollama.
- Create a state graph to enable short-term memory using LangGraph.
- Build the agent app for chat conversation.
- Ask the same set of queries for each model.
- Compare time and output text quality.

# Reference document feeding to local RAG
https://www.biosciences-labs.bham.ac.uk/filatov/cv.txt

# Open-source LLMs used for LangChain-Ollama

- gemma3:1b
- qwen3:1.7b
- llama3.1
- gemma3:4b
- cogito:3b
- cogito:8b
- deepseek-r1:1.5b
- deepseek-r1:8b

# Queries to ask in local RAG (for each model)
- Which university did he or she get Ph.D.?
- Which countries did he or she work afterwards?
- Apart from English, what other languages does he or she speak?

# Comparison Result
<p align="center"><img src="https://github.com/user-attachments/assets/1f62dc71-768c-415a-b736-fa6cdcc7f7a5" width="700"/></p>

# Answers to the queries by each model

-------------------------------------------------
By using gemma3:1b, the LLM generates
-------------------------------------------------
He got his Ph. D. at the Institute of Molecular Genetics Moscow, Russia.
After his Ph.D., he worked at the following locations:

*   **Russia** (Institute of Molecular Genetics, Moscow)
*   **United Kingdom** (University of Edinburgh, University of Birmingham, University of Oxford)
*   **France** (University of Lyon, EMBO short term fellowship)
*   **Scotland** (University of Dundee)
*   **Netherlands** (University of Wageningen)
*   **Denmark** (University of Aarhus)
*   **Paris, France** ( EMBO short term fellowship)
*   **United States** (University of California, Davis)
  
Besides native Russian and fluent English, Dmitry A. Filatov speaks:

*   French
  
-------------------------------------------------
By using gemma3:1b, it takes 0:00:14 to finish.
-------------------------------------------------

-------------------------------------------------
By using qwen3:1.7b, the LLM generates:
-------------------------------------------------
<think>
Okay, let's see. The user is asking which university he got his Ph.D. from. I need to check the provided context to find that information.

Looking through the research experience section, there's a mention of his PhD work at the Institute of Molecular Genetics Moscow, Russia. The context says, "Research work (project student, then PhD student) at the Institute of Molecular Genetics Moscow, Russia." So the university is the Institute of Molecular Genetics in Moscow. 

I should make sure there's no other university mentioned. The other entries are about teaching experiences, employment, and seminars, but not about his PhD institution. So the answer should be the Institute of Molecular Genetics in Moscow, Russia.
</think>

The individual obtained his Ph.D. from the Institute of Molecular Genetics in Moscow, Russia.
<think>
Okay, the user is asking which countries he worked afterwards. Let me check the provided information.

Looking at the employment section, after 2001, he's a lecturer at the University of Birmingham. Then, under research experience, there's a mention of his postdoctoral work at the University of Edinburgh from 1998-2001. Before that, in 1993-1998, he was a research student at the Institute of Molecular Genetics in Moscow. Then there were research visits to Lyon, Dundee, and others. But the question is about "afterwards," so after his postdoc at Edinburgh. 

Wait, the user might be referring to all the countries he worked in. Let me list them out. Moscow for his PhD, then Edinburgh for postdoc, and then the University of Birmingham as his current position. Also, he had research visits to France, Scotland, and the US. So the countries are Russia, UK (Edinburgh and Birmingham), France, Scotland, and the US. But the user might want a concise list. Let me make sure I didn't miss any. The answer should include all the countries he worked in after his PhD. So Moscow, then Edinburgh,
<think>
Okay, the user is asking about the languages the person speaks apart from English. Let me check the provided information.

In the "Languages" section, it says: "native Russian, fluent English, basic French." So, besides English, he speaks Russian and French. The basic French part means he's fluent in French, but not necessarily advanced. So the answer should list Russian and French as the additional languages. I need to make sure there's no other language mentioned. The text doesn't mention any other languages, so the answer is Russian and French.
</think>

The individual is fluent in English and native to Russian, with basic French proficiency.

-------------------------------------------------
By using qwen3:1.7b, it takes 0:00:25 to finish.
-------------------------------------------------

-------------------------------------------------
By using llama3.1, the LLM generates :
-------------------------------------------------
The context does not explicitly mention where the person got their PhD from, but it does mention that they worked as a project student and then PhD student at the Institute of Molecular Genetics Moscow, Russia from 1993-1998. So, it can be inferred that the person likely received their PhD from this institution.
According to the context, after working in Russia (Institute of Molecular Genetics Moscow), the person worked in:

1. France (University of Lyon) for a research visit funded by a fellowship from the French Ministry of Education.
2. Scotland (University of Dundee) for a research visit funded by an EMBO short term fellowship.
3. United Kingdom (University of Edinburgh, and then at the University of Birmingham) as a postdoctoral research fellow and later as a lecturer in evolutionary genetics.

According to the context, apart from his native Russian, he also speaks fluent English and has basic French language skills.

-------------------------------------------------
By using llama3.1, it takes 0:01:24 to finish.
-------------------------------------------------

-------------------------------------------------
By using gemma3:4b, the LLM generates :
-------------------------------------------------
He got his Ph.D. at the Institute of Molecular Genetics Moscow, Russia.
Based on the provided text, he worked in the following countries after his PhD:

*   United Kingdom (University of Edinburgh, University of Birmingham)
*   Russia (Institute of Molecular Genetics Moscow)
*   France (University of Lyon)
*   Scotland (University of Dundee)
*   Netherlands (University of Wageningen)
*   Denmark (University of Aarhus)
*   United Kingdom (University of Oxford, University College London)
*   United Kingdom (The Sanger Center, Cambridge)

He speaks native Russian and fluent English, with basic French.

-------------------------------------------------
By using gemma3:4b, it takes 0:02:08 to finish.
-------------------------------------------------

-------------------------------------------------
By using cogito:3b, the LLM generates :
-------------------------------------------------
I don't know.

While the context mentions that during his PhD he worked as an evolutionary geneticist, it doesn't specify which university he got his Ph.D. from. The only universities mentioned in his employment and research experience are University of Birmingham (2001-present), University of Edinburgh (1998-2001), and others where he did research visits or had collaborations.

So without more specific information, I can't determine the exact university where he completed his Ph.D.
I don't know.

Based on the provided context, we have information about his employment and research activities at:

1. University of Birmingham (UK) - From 2001 to present
2. University of Edinburgh (UK) - From 1998 to 2001
3. Research visits in France (University of Lyon), Scotland (University of Dundee)

However, the text doesn't explicitly mention any countries where he worked after his Ph.D. The only country mentioned in his research experience is Russia (Institute of Molecular Genetics Moscow). 

So without more specific information about his post-Ph.D. employment or research activities, I can't determine which countries he worked afterwards.
According to the provided context:

He speaks:
1. Native Russian
2. Fluent English
3. Basic French


So apart from English, he also knows Russian and basic French.

-------------------------------------------------
By using cogito:3b, it takes 0:00:39 to finish.
-------------------------------------------------

-------------------------------------------------
By using cogito:8b, the LLM generates :
-------------------------------------------------
Based on the provided context, it appears that he received his PhD at the Institute of Molecular Genetics Moscow, Russia, where he worked as a research project student and then PhD student from 1993 to 1998. However, there is no explicit mention in the text of which university awarded him his PhD degree.

Therefore, I don't know for certain which university awarded him his Ph.D.
Based on the provided context, here are the countries where he worked after receiving his PhD:

1. Russia (1993-1998) - At the Institute of Molecular Genetics Moscow
2. France (November 1996 - April 1997) - Research visit at University of Lyon
3. Scotland (September 1997- October 1997) - Research visit at University of Dundee
4. UK (1998-2001) - Postdoctoral research fellow at University of Edinburgh
5. UK (2001-present) - Lecturer in Evolutionary Genetics at the University of Birmingham


According to the provided context, apart from English, he speaks:

- Native Russian
- Basic French

-------------------------------------------------
By using cogito:8b, it takes 0:01:26 to finish.
-------------------------------------------------

-------------------------------------------------
By using deepseek-r1:1.5b, the LLM generates :
-------------------------------------------------
<think>
Okay, so I need to figure out which university Dr. Alex Smith got his PhD from based on the information provided. Let me go through this step by step.

First, looking at his research experience: He worked as a postdoctoral researcher at the University of Edinburgh from 1993 to 1998. That's pretty clear about where he did his Ph.D., but I should make sure there are no other details that might change this.

I don't see any information about his education or the university where he received his PhD mentioned in the provided context. The only places mentioned related to his research are the Russian native language, fluent English, basic French background, and various invited seminars and meetings.

So, unless there's additional information elsewhere that I'm missing, it seems like the University of Edinburgh is the most likely candidate for where he got his PhD based on his postdoctoral work there. 

I should double-check if there are any other universities or institutions mentioned in the context that could be relevant, but from what's provided, the only connection to a university with postdoctoral experience is the University of Edinburgh.

Therefore, I'm confident that Dr. Alex Smith earned his PhD at the University of Edinburgh.
</think>

Dr. Alex
<think>
Okay, so the user is asking about which countries Dr. Alex worked after his PhD. From the context provided, I know that Dr. Alex was a lecturer in Evolutionary Genetics at the University of Birmingham from 2001 to present. He also had a postdoctoral research fellow position at the University of Edinburgh from 1993-1998.

I should check both his PhD and postdoc experiences. The user's previous question about countries was answered, but now they're asking specifically about the years after his PhD.

Looking up Dr. Alex's work experience after his PhD: He worked at the University of Edinburgh for a few years as a postdoc. Then, moving from Birmingham to Edinburgh, he was involved in developing software for evolutionary analysis and creating a population genetics database. 

I need to make sure I include both the UK and the USA aspects since he was based there during his time at Edinburgh.

So, putting it all together: Dr. Alex worked in the United Kingdom as a postdoctoral fellow at the University of Edinburgh and later moved to the USA, working at the Sanger Center in Cambridge.
</think>

Dr. Alex worked in the United Kingdom as a postdoctoral fellow at the University of Edinburgh, where
<think>
Okay, let me try to figure out how to approach this question. The user is asking about Dr. Alex's languages besides English. From the context provided earlier, I know that Dr. Alex has a native Russian and is fluent in English and basic French.

I remember seeing that his hobby includes programming with databases and web programming using languages like Delphi, Perl, Java Script. So, he uses multiple languages for different tasks. Additionally, he mentions being employed at the University of Birmingham, which suggests they might have some English proficiency as well.

Wait, but the user specifically asked about other languages besides English. So, I should focus on Russian and French since those are the languages he is fluent in. Maybe also mention his programming skills if applicable, even though it's not strictly a language.

I need to make sure I'm accurate without adding any personal opinions or assumptions. The information given is clear: he has native Russian, fluencies in English and basic French, and uses multiple programming languages for different projects.
</think>

Dr. Alex primarily speaks Russian as his native language. He is also fluent in English and can speak a bit of basic French. Additionally, he may be familiar with programming languages used in his research or work experience.

-------------------------------------------------
By using deepseek-r1:1.5b, it takes 0:00:32 to finish.
-------------------------------------------------

-------------------------------------------------
By using deepseek-r1:8b, the LLM generates :
-------------------------------------------------
<think>
Alright, I need to figure out which university the person got their PhD from based on the given information. Let me read through the context carefully.

The user provided a detailed background about someone with a PhD in molecular genetics. The employment section mentions that they were a postdoctoral research fellow at the University of Edinburgh from 1998 to 2001, working under Prof. D. Charlesworth. That suggests their PhD might be from before 1998.

Looking further into the research experience, it says they worked as a project student and then PhD student at the Institute of Molecular Genetics in Moscow, Russia between 1993-1998. So that's where their PhD likely took place. They also had visits to other universities, but those were after their PhD.

So putting it together, the person received their PhD from the Institute of Molecular Genetics, which is part of the University of Moscow.
</think>

The individual earned their PhD at the Institute of Molecular Genetics, which is affiliated with the University of Moscow. 

**Answer:** The University of Moscow.
<think>
Alright, so I need to figure out which countries the person worked in after getting his PhD. Let me start by looking through the information provided.

First, from the background, it says that they completed their PhD at the Institute of Molecular Genetics, which is part of the University of Moscow. So that's Russia.

Then, under research experience, there are a few points. From November 1996 to April 1997, they were at the University of Lyon in France on a fellowship from the French Ministry of Education. That's another country, France.

Next, September 1997 to October 1997, they visited the University of Dundee in Scotland with an EMBO short-term fellowship. So that adds Scotland, which is part of the UK.

From 1998 to 2001, they worked as a postdoctoral research fellow in Professor David Charlesworth's lab at the University of Edinburgh, UK. That's another UK location, specifically Scotland again but a different institution.

After 2001, they moved to an independent position and have been working at the University of Birmingham since then, which is also in the UK.

Additionally, there are mentions of invited seminars at various places, including the University of Bath (UK), University of Paris (
<think>
Okay, so I need to figure out what other languages the person speaks besides English. Let me go through the information provided.

From the languages section, it says they are native in Russian and fluent in English. It also mentions basic French. So that's two languages: Russian and English, with some proficiency in French.

I don't see any other languages mentioned elsewhere in the context provided. There's no mention of other languages in their employment history or research experience. They might know more languages, but based on what's given, only Russian, English, and basic French are noted.

So, to answer the question, they speak Russian (native), English (fluent), and have basic proficiency in French.
</think>

The person speaks Russian (native) and English (fluent), with some knowledge of French.

-------------------------------------------------
By using deepseek-r1:8b, it takes 0:04:33 to finish.
-------------------------------------------------
