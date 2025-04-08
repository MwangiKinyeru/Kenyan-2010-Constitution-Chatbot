import logging
import json
import spacy
from spellchecker import SpellChecker
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import nest_asyncio
import asyncio
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_file(filename):
    """Helper function to load JSON files with error handling"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            logger.info(f"Successfully loaded {filename}")
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filename}")
        return {}
    except Exception as e:
        logger.error(f"Error loading {filename}: {str(e)}")
        return {}

# Load environment variables
load_dotenv()
bot_token = os.getenv("BOT_TOKEN")
if not bot_token:
    logger.critical("BOT_TOKEN not found in environment variables")
    exit(1)
logger.info("Environment variables loaded successfully")

# Apply the nest_asyncio patch
nest_asyncio.apply()

try:
    # Initialize NLP components
    logger.info("Loading NLP models...")
    nlp = spacy.load("en_core_web_sm")
    spell = SpellChecker()
    logger.info("NLP models loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load NLP models: {str(e)}")
    exit(1)

# Load knowledge base files
sections = load_json_file("combined_sections.json")
synonyms = load_json_file("synonyms.json")
qa_mapping = load_json_file("qa_mapping.json")
citizenship_mapping = load_json_file("citizenship_mapping.json")

def preprocess_query(query):
    """Clean and normalize user query"""
    try:
        doc = nlp(query)
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and not token.is_space]
        processed = " ".join(tokens)
        logger.debug(f"Preprocessed query: '{query}' -> '{processed}'")
        return processed
    except Exception as e:
        logger.error(f"Error preprocessing query: {str(e)}")
        return query.lower()  # Fallback to simple lowercase

def correct_spelling(processed_query):
    """Correct spelling in processed query"""
    try:
        words = processed_query.split()
        misspelled = spell.unknown(words)
        corrected = [spell.correction(word) if word in misspelled else word 
                    for word in words]
        result = " ".join(corrected)
        logger.debug(f"Spell correction: '{processed_query}' -> '{result}'")
        return result
    except Exception as e:
        logger.error(f"Error in spell correction: {str(e)}")
        return processed_query

def match_with_synonyms(query, qa_mapping, synonyms, citizenship_mapping):
    """Find matching section using synonym expansion"""
    processed_query = preprocess_query(query)
    corrected_query = correct_spelling(processed_query)
    
    logger.debug(f"Match attempt: Original='{query}' Processed='{processed_query}' Corrected='{corrected_query}'")

    # Check citizenship topics first
    for subtopic, section_key in citizenship_mapping.items():
        if subtopic.lower() in corrected_query:
            logger.debug(f"Matched citizenship subtopic: {subtopic}")
            return section_key

    if "citizenship" in corrected_query:
        logger.debug("Matched general citizenship topic")
        return "citizenship"

    # Check against QA mapping with synonyms
    for key in qa_mapping:
        # Check both main key and all synonyms
        search_terms = [key.lower()] + [s.lower() for s in synonyms.get(key, [])]
        for term in search_terms:
            if term in corrected_query:
                logger.debug(f"Matched term: {term} (original key: {key})")
                return key
                
    logger.debug("No matches found in synonym mapping")
    return None

def answer_question_nlp(query):
    """Main function to generate answers to constitutional questions"""
    try:
        logger.info(f"Processing query: '{query}'")
        
        # Check prioritized multi-word phrases first
        prioritized_phrases = {
            ('language', 'culture'): 'language culture',
            ('implementation', 'rights'): 'implementation right',
            ('authority', 'court', 'bill', 'right'): 'authority court bill right',
            # ... (keep your existing prioritized phrases)
        }

        query_lower = query.lower()
        for terms, key in prioritized_phrases.items():
            if all(term in query_lower for term in terms):
                logger.debug(f"Matched prioritized phrase: {terms}")
                if key in sections:
                    return sections[key]

        # Fallback to synonym matching
        section_key = match_with_synonyms(query, qa_mapping, synonyms, citizenship_mapping)
        
        if section_key in sections:
            return sections[section_key]
        elif section_key == "citizenship":
            return (f"It seems you're interested in citizenship. "
                   f"Available subtopics include: {list(citizenship_mapping.keys())}.")
            
        return ("Sorry, I couldn't find an answer to your question. "
               "Try asking about specific topics like:\n"
               "- Citizenship requirements\n"
               "- Parliamentary authority\n"
               "- Constitutional amendments")
                
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return "Sorry, I encountered an error processing your question."

async def handle_message(update: Update, context: CallbackContext) -> None:
    """Handle incoming Telegram messages"""
    try:
        if not update.message or not update.message.text:
            logger.warning("Received empty message or non-text update")
            return
            
        user_query = update.message.text
        logger.info(f"Received message from {update.effective_user.id}: {user_query}")
        
        # Show typing indicator while processing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        answer = answer_question_nlp(user_query)
        await update.message.reply_text(answer)
        
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        if update.message:
            await update.message.reply_text(
                "Sorry, I encountered an error. Please try again later."
            )

async def error_handler(update: object, context: CallbackContext) -> None:
    """Handle errors in the telegram bot"""
    logger.error(f"Update {update} caused error: {context.error}")
    if update and hasattr(update, 'message'):
        await update.message.reply_text(
            "An error occurred. The developers have been notified."
        )

async def main() -> None:
    """Start the bot"""
    try:
        logger.info("Starting bot...")
        application = (
            ApplicationBuilder()
            .token(bot_token)
            .build()
        )
        
        # Add handlers
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_error_handler(error_handler)
        
        # Start polling
        logger.info("Bot started, waiting for messages...")
        await application.run_polling()
        
    except Exception as e:
        logger.critical(f"Failed to start bot: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")