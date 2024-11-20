package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/joho/godotenv"
	"github.com/meilisearch/meilisearch-go"
	"github.com/sashabaranov/go-openai"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// Config estructura para configuraciones globales
type Config struct {
	MaxConcurrentDocuments   int
	MaxConcurrentExtractions int
	ExtractionTimeout        time.Duration
	ShutdownTimeout          time.Duration
}

type ExtractorClient struct {
	client  *openai.Client
	baseURL string
	name    string
}

// Primero, creamos una estructura para manejar múltiples clientes OpenAI
type LoadBalancedOpenAIClient struct {
	extractors    []*ExtractorClient
	currentIndex  int32
	logger        *Logger
	healthChecks  map[string]bool
	healthMutex   sync.RWMutex
	checkInterval time.Duration
}

// ANSI color codes
const (
	Reset    = "\033[0m"
	Bold     = "\033[1m"
	Red      = "\033[31m"
	Green    = "\033[32m"
	Yellow   = "\033[33m"
	Blue     = "\033[34m"
	Magenta  = "\033[35m"
	Cyan     = "\033[36m"
	White    = "\033[37m"
	BgRed    = "\033[41m"
	BgGreen  = "\033[42m"
	BgYellow = "\033[43m"
	BgBlue   = "\033[44m"
)

type Logger struct {
	prefix string
}

func NewLogger(prefix string) *Logger {
	return &Logger{prefix: prefix}
}

func (l *Logger) formatMessage(level, color, message string) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	return fmt.Sprintf("%s%s %s[%s]%s %s[%s]%s %s",
		Bold, timestamp,
		color, level, Reset,
		Cyan, l.prefix, Reset,
		message)
}

func (l *Logger) Info(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Println(l.formatMessage("INFO", Green, msg))
}

func (l *Logger) Warning(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Println(l.formatMessage("WARN", Yellow, msg))
}

func (l *Logger) Error(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Println(l.formatMessage("ERROR", Red, msg))
}

func (l *Logger) Debug(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Println(l.formatMessage("DEBUG", Blue, msg))
}

func (l *Logger) Success(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Println(l.formatMessage("SUCCESS", Green, Bold+msg+Reset))
}

func (l *Logger) Critical(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Println(l.formatMessage("CRITICAL", BgRed+White, msg))
}

// OpenAIClient handles LLM operations
type OpenAIClient struct {
	Extractor       *openai.Client
	ModelsExtractor openai.ModelsList
	logger          *Logger
}

func NewOpenAIClient() *OpenAIClient {
	logger := NewLogger("OpenAIClient")
	logger.Info("Initializing OpenAI client...")

	llmKey := os.Getenv("NEBUIA_LLM_KEY")
	extractorConfig := openai.DefaultConfig(llmKey)
	extractorServer := os.Getenv("EXTRACTOR_SERVER")
	extractorConfig.BaseURL = extractorServer
	extractorConfig.HTTPClient = &http.Client{
		Timeout: 20 * time.Second,
	}

	extractorClient := openai.NewClientWithConfig(extractorConfig)
	modelsExtractor, err := extractorClient.ListModels(context.Background())
	if err != nil {
		logger.Warning("Failed to list models: %v", err)
	}

	logger.Success("OpenAI client initialized successfully")
	return &OpenAIClient{
		Extractor:       extractorClient,
		ModelsExtractor: modelsExtractor,
		logger:          logger,
	}
}

func NewLoadBalancedOpenAIClient() *LoadBalancedOpenAIClient {
	logger := NewLogger("LoadBalancedLLM")
	logger.Info("Initializing load balanced OpenAI client...")

	llmKey := os.Getenv("NEBUIA_LLM_KEY")

	serverConfigs := []struct {
		name string
		url  string
	}{
		{"Extractor1", "https://extractor1.nebuia.com/v1"},
		{"Extractor2", "https://extractor.nebuia.com/v1"},
		// {"Extractor3", "https://extractor3.nebuia.com/v1"},
	}

	extractors := make([]*ExtractorClient, len(serverConfigs))
	healthChecks := make(map[string]bool)

	for i, server := range serverConfigs {
		config := openai.DefaultConfig(llmKey)
		config.BaseURL = server.url
		config.HTTPClient = &http.Client{
			Timeout: 20 * time.Second,
		}

		extractors[i] = &ExtractorClient{
			client:  openai.NewClientWithConfig(config),
			baseURL: server.url,
			name:    server.name,
		}
		healthChecks[server.url] = true
		logger.Info("Registered server %s: %s", server.name, server.url)
	}

	client := &LoadBalancedOpenAIClient{
		extractors:    extractors,
		currentIndex:  -1,
		logger:        logger,
		healthChecks:  healthChecks,
		checkInterval: 30 * time.Second,
	}

	// Iniciar health checks
	go client.startHealthChecks()

	logger.Success("Load balanced OpenAI client initialized with %d servers", len(extractors))
	return client
}

func (lb *LoadBalancedOpenAIClient) startHealthChecks() {
	ticker := time.NewTicker(lb.checkInterval)
	for range ticker.C {
		lb.checkHealth()
	}
}

func (lb *LoadBalancedOpenAIClient) checkHealth() {
	for _, extractor := range lb.extractors {
		go func(e *ExtractorClient) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			_, err := e.client.ListModels(ctx)

			lb.healthMutex.Lock()
			defer lb.healthMutex.Unlock()

			if err != nil {
				if lb.healthChecks[e.baseURL] {
					lb.logger.Error("Error health %s", err.Error())
					lb.logger.Warning("Server %s is down: %v", e.name, err)
				}
				lb.healthChecks[e.baseURL] = false
			} else {
				if !lb.healthChecks[e.baseURL] {
					lb.logger.Success("Server %s is back online", e.name)
				}
				lb.healthChecks[e.baseURL] = true
			}
		}(extractor)
	}
}

func (lb *LoadBalancedOpenAIClient) getNextHealthyClient() *ExtractorClient {
	lb.healthMutex.RLock()
	defer lb.healthMutex.RUnlock()

	numServers := int32(len(lb.extractors))
	startIndex := atomic.AddInt32(&lb.currentIndex, 1) % numServers

	// Intentar con todos los servidores empezando por el siguiente
	for i := int32(0); i < numServers; i++ {
		index := (startIndex + i) % numServers
		extractor := lb.extractors[index]
		if lb.healthChecks[extractor.baseURL] {
			//lb.logger.Debug("Selected server %s (%d of %d)",
			//	extractor.name, index+1, numServers)
			return extractor
		}
	}

	// Si ninguno está saludable, usar el siguiente en la lista
	extractor := lb.extractors[startIndex]
	lb.logger.Warning("No healthy servers available, using %s", extractor.name)
	return extractor
}

func predictExtract(text, schema string) (string, error) {
	var schemaObj interface{}
	err := json.Unmarshal([]byte(schema), &schemaObj)
	if err != nil {
		return "", fmt.Errorf("error unmarshaling schema: %v", err)
	}

	formattedSchema, err := json.MarshalIndent(schemaObj, "", "    ")
	if err != nil {
		return "", fmt.Errorf("error marshaling schema: %v", err)
	}

	inputLLM := "<|input|>\n### Template:\n" + string(formattedSchema) + "\n"
	inputLLM += "### Text:\n" + text + "\n<|output|>\n"

	return inputLLM, nil
}

// Función auxiliar para obtener los primeros n caracteres de un string
func firstN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// SearchService handles search operations
type SearchService struct {
	client meilisearch.ServiceManager
	logger *Logger
}

func InitMeiliSearch() meilisearch.ServiceManager {
	logger := NewLogger("MeiliSearch")
	logger.Info("Initializing MeiliSearch client...")
	client := meilisearch.New(os.Getenv("MEILI_HOST"), meilisearch.WithAPIKey(os.Getenv("MEILI_KEY")))
	logger.Success("MeiliSearch client initialized")
	return client
}

type SearchRequest struct {
	UUID       string
	Matches    string
	MaxResults int64
}

func (s *SearchService) Search(req SearchRequest) (*meilisearch.SearchResponse, error) {
	//s.logger.Info("Searching in index %s with query: %s", req.UUID, req.Matches)
	searchRes, err := s.client.Index(req.UUID).Search(req.Matches,
		&meilisearch.SearchRequest{
			Limit: req.MaxResults,
		})
	if err != nil {
		s.logger.Error("Search failed: %v", err)
		return nil, fmt.Errorf("error searching in index: %v", err)
	}
	//s.logger.Success("Search completed successfully with %d results", len(searchRes.Hits))
	return searchRes, nil
}

// FileRepository handles MongoDB operations
type FileRepository struct {
	db        *mongo.Database
	batches   *mongo.Collection
	documents *mongo.Collection
	logger    *Logger
}

func NewFileRepository(db *mongo.Database) *FileRepository {
	return &FileRepository{
		db:        db,
		batches:   db.Collection("batches"),
		documents: db.Collection("documents"),
		logger:    NewLogger("FileRepository"),
	}
}

func (fr *FileRepository) SetDocumentReviewStatus(status, uuid string) error {
	fr.logger.Info("Setting document %s status to: %s", uuid, status)
	filter := bson.M{"uuid": uuid}
	update := bson.M{
		"$set": bson.M{
			"status_document": status,
		},
	}

	result, err := fr.documents.UpdateOne(context.Background(), filter, update)
	if err != nil {
		fr.logger.Error("Failed to update document status: %v", err)
		return fmt.Errorf("error on document update: %v", err)
	}
	if result.ModifiedCount == 0 {
		fr.logger.Warning("No document was updated for UUID: %s", uuid)
		return fmt.Errorf("document not found for update")
	}
	fr.logger.Success("Document status updated successfully")
	return nil
}

func (fr *FileRepository) AddEntitiesToDocumentInBatch(uuid string, entities map[string]interface{}) error {
	//fr.logger.Info("Adding/Updating entities for document UUID: %s with data: %+v", uuid, entities)

	// Verificamos si el documento existe
	var doc bson.M
	err := fr.documents.FindOne(context.Background(), bson.M{"uuid": uuid}).Decode(&doc)
	if err != nil {
		if err == mongo.ErrNoDocuments {
			return fmt.Errorf("document with UUID %s not found in database", uuid)
		}
		return fmt.Errorf("error checking document existence: %v", uuid)
	}

	// Primero eliminamos la entidad específica si existe
	pullFilter := bson.M{
		"uuid": uuid,
	}
	pullUpdate := bson.M{
		"$pull": bson.M{
			"entities": bson.M{
				"key":     entities["key"],
				"id_core": entities["id_core"],
			},
		},
	}

	// Eliminar la entidad existente
	_, err = fr.documents.UpdateOne(context.Background(), pullFilter, pullUpdate)
	if err != nil {
		fr.logger.Error("Failed to remove existing entity: %v", err)
		return fmt.Errorf("error removing existing entity: %v", err)
	}

	// Ahora agregamos la nueva entidad
	pushFilter := bson.M{"uuid": uuid}
	pushUpdate := bson.M{
		"$push": bson.M{
			"entities": entities,
		},
	}

	result, err := fr.documents.UpdateOne(context.Background(), pushFilter, pushUpdate)
	if err != nil {
		fr.logger.Error("Failed to add new entity: %v", err)
		return fmt.Errorf("database error while adding entity: %v", err)
	}

	if result.ModifiedCount == 0 {
		fr.logger.Error("Document %s was found but entity was not added", uuid)
		return fmt.Errorf("document found but entity not added - UUID: %s", uuid)
	}

	//fr.logger.Success("Successfully updated entity for document %s", uuid)
	return nil
}

func (fr *FileRepository) GetDocumentsFromBatch(batchID string) ([]bson.M, error) {
	fr.logger.Info("Retrieving 'in_list_llm' documents for batch: %s", batchID)
	objID, err := primitive.ObjectIDFromHex(batchID)
	if err != nil {
		fr.logger.Error("Invalid batch ID format: %v", err)
		return nil, err
	}

	filter := bson.M{
		"batch_id": objID,
	}

	cursor, err := fr.documents.Find(context.Background(), filter)
	if err != nil {
		fr.logger.Error("Failed to find documents: %v", err)
		return nil, err
	}
	defer cursor.Close(context.Background())

	var allResults []bson.M
	if err = cursor.All(context.Background(), &allResults); err != nil {
		fr.logger.Error("Failed to decode documents: %v", err)
		return nil, err
	}

	// Filtrar solo documentos con status_document = "in_list_llm"
	var filteredResults []bson.M
	for _, doc := range allResults {
		if status, ok := doc["status_document"].(string); ok && status == "in_list_llm" {
			filteredResults = append(filteredResults, doc)
		}
	}

	fr.logger.Success("Retrieved %d 'in_list_llm' documents from batch (from total %d)",
		len(filteredResults), len(allResults))

	if len(filteredResults) == 0 {
		fr.logger.Warning("No documents with status 'in_list_llm' found in batch")
	}

	return filteredResults, nil
}

// PipelineRepository handles pipeline operations
type PipelineRepository struct {
	collection *mongo.Collection
	logger     *Logger
}

func NewPipelineRepository(db *mongo.Database) *PipelineRepository {
	return &PipelineRepository{
		collection: db.Collection("pipelines"),
		logger:     NewLogger("PipelineRepo"),
	}
}

func (pr *PipelineRepository) GetPipelinesByApplyTo(idToFind string) (bson.M, error) {
	pr.logger.Info("Looking for pipelines with ID: %s", idToFind)
	objID, err := primitive.ObjectIDFromHex(idToFind)
	if err != nil {
		pr.logger.Error("Invalid ID format: %v", err)
		return nil, err
	}

	var pipeline bson.M
	err = pr.collection.FindOne(context.Background(), bson.M{"apply_to": objID}).Decode(&pipeline)
	if err != nil {
		pr.logger.Error("Failed to find pipeline: %v", err)
		return nil, err
	}
	pr.logger.Success("Pipeline found successfully")
	return pipeline, nil
}

// FileService handles file operations and pipeline processing
type FileService struct {
	repo                *FileRepository
	pipe                *PipelineRepository
	search              *SearchService
	source              *redis.Client
	openAIClient        *LoadBalancedOpenAIClient
	logger              *Logger
	config              Config
	extractionSemaphore chan struct{}
	ctx                 context.Context
	cancel              context.CancelFunc
}

func NewFileService(repo *FileRepository, pipe *PipelineRepository, searchClient meilisearch.ServiceManager) *FileService {
	ctx, cancel := context.WithCancel(context.Background())
	config := Config{
		MaxConcurrentDocuments:   10,
		MaxConcurrentExtractions: 200,
		ExtractionTimeout:        30 * time.Second,
		ShutdownTimeout:          60 * time.Second,
	}

	return &FileService{
		repo:         repo,
		pipe:         pipe,
		search:       &SearchService{client: searchClient, logger: NewLogger("SearchService")},
		openAIClient: NewLoadBalancedOpenAIClient(),
		source: redis.NewClient(&redis.Options{
			Addr:     fmt.Sprintf("%s:%s", os.Getenv("REDIS_HOST"), os.Getenv("REDIS_PORT")),
			Password: os.Getenv("REDIS_PASSWORD"),
			DB:       0,
		}),
		logger:              NewLogger("FileService"),
		config:              config,
		extractionSemaphore: make(chan struct{}, config.MaxConcurrentExtractions),
		ctx:                 ctx,
		cancel:              cancel,
	}
}

func (fs *FileService) Shutdown() {
	fs.logger.Info("Initiating FileService shutdown...")
	fs.cancel()

	// Esperar un tiempo razonable para que se completen las operaciones en curso
	timer := time.NewTimer(fs.config.ShutdownTimeout)
	<-timer.C
	fs.logger.Success("FileService shutdown complete")
}

func (fs *FileService) RemoveItem(batchID string) error {
	fs.logger.Info("Removing batch: %s", batchID)
	count, err := fs.source.LRem(context.Background(), "to_operate", 0, batchID).Result()
	if err != nil {
		fs.logger.Error("Failed to remove batch: %v", err)
		return err
	}
	fs.logger.Success("Removed %d occurrences of batch '%s'", count, batchID)
	return nil
}

func (fs *FileService) GetItemToWork() (string, error) {
	//fs.logger.Debug("Fetching next item to work on")
	batchID, err := fs.source.LIndex(context.Background(), "to_operate", -1).Result()
	if err != nil {
		if err == redis.Nil {
			//fs.logger.Info("No items found in queue")
			return "", nil
		}
		fs.logger.Warning("Failed to get next item: %v", err)
		return "", err
	}
	fs.logger.Success("Found batch to process: %s", batchID)
	return batchID, nil
}

func (fs *FileService) ProcessPipelineToDocument(ctx context.Context, pipe bson.M, doc bson.M) (string, int, error) {
	select {
	case fs.extractionSemaphore <- struct{}{}: // Adquirir un slot para extracción
		defer func() { <-fs.extractionSemaphore }() // Liberar el slot al terminar
	case <-ctx.Done():
		return "", -1, ctx.Err()
	}

	//fs.logger.Info("Processing pipeline for document: %s", doc["uuid"])
	if pipe["value"] == nil || pipe["key"] == nil || pipe["value"] == "" || pipe["key"] == "" {
		//fs.logger.Warning("No instructions found in pipeline")
		return "sin instrucción", -1, nil
	}

	searchReq := SearchRequest{
		UUID:       doc["uuid"].(string),
		Matches:    pipe["match"].(string),
		MaxResults: 1,
	}

	/*fs.logger.Debug(`
	🔍 Search Request Details:
	UUID: %s
	Match Query: %s
	Pipeline Name: %s
	`, searchReq.UUID, searchReq.Matches, pipe["name"]) */

	searchRes, err := fs.search.Search(searchReq)
	if err != nil || len(searchRes.Hits) == 0 {
		fs.logger.Warning(`
⚠️ No Search Results
Document ID: %s
Pipeline Name: %s
Match Query: %s
Error: %v
`, doc["uuid"], pipe["name"], searchReq.Matches, err)
		return "no_encontrado", -1, nil
	}

	useLlama := false
	if llamaVal, exists := pipe["use_llama"]; exists {
		if boolVal, ok := llamaVal.(bool); ok {
			useLlama = boolVal
		}
	}

	if !useLlama {
		hits := searchRes.Hits
		hit0 := hits[0].(map[string]interface{})
		meta := hit0["meta"].(map[string]interface{})

		var page int
		switch sourceVal := meta["source"].(type) {
		case float64:
			page = int(sourceVal)
		case int:
			page = sourceVal
		case int64:
			page = int(sourceVal)
		default:
			fs.logger.Warning(`
⚠️ Unexpected Page Type
Document ID: %s
Pipeline Name: %s
Type Found: %T
Value: %v
`, doc["uuid"], pipe["name"], meta["source"], meta["source"])
			page = -1
		}

		combinedContent := hit0["content"].(string)
		/*if len(hits) > 1 {
			if hit1, ok := hits[1].(map[string]interface{}); ok {
				if content, ok := hit1["content"].(string); ok {
					combinedContent += "\n" + content
				}
			}
		}*/

		schema := generateSchema(pipe)
		/*fs.logger.Debug(`
		🔧 Processing Details:
		Document ID: %s
		Pipeline Name: %s
		Page: %d
		Schema: %s
		Content Length: %d characters
		`, doc["uuid"], pipe["name"], page, schema, len(combinedContent)) */

		extractCtx, cancel := context.WithTimeout(ctx, fs.config.ExtractionTimeout)
		defer cancel()

		response, err := fs.openAIClient.ExtractorInference(extractCtx, combinedContent, schema)
		if err != nil {
			fs.logger.Error(`
🔴 Extraction Error
Pipeline Name: %s
Document ID: %s
Error: %v
Schema:
%s
Content Preview (first 500 chars):
%s
`, pipe["name"], doc["uuid"], err, schema, firstN(combinedContent, 500))
			return "error_en_extraccion", -1, err
		}

		var result map[string]interface{}
		if err := json.Unmarshal([]byte(*response), &result); err != nil {
			fs.logger.Error(`
🔴 JSON Parsing Error
Pipeline Name: %s
Document ID: %s
Error: %v
Response:
%s
Schema:
%s
`, pipe["name"], doc["uuid"], err, *response, schema)
			return "error_en_extraccion", -1, err
		}

		value := extractValue(result)
		if _, ok := value.(map[string]interface{}); !ok {
			/*fs.logger.Success(`
			✅ Extraction Completed
			Pipeline Name: %s
			Document ID: %s
			Page: %d
			Result: %v
			`, pipe["name"], doc["uuid"], page, value)
						return fmt.Sprint(value), page, nil */
			return fmt.Sprint(value), page, nil
		} else {
			fs.logger.Error(`
🔴 JSON Marshal Error
Pipeline Name: %s
Document ID: %s
Error: %v
Result: %+v
`, pipe["name"], doc["uuid"], err, result)
			return "error_en_extraccion", -1, err
		}
	}

	/*jsonBytes, err := json.Marshal(result)
			if err != nil {
				fs.logger.Error(`
	🔴 JSON Marshal Error
	Pipeline Name: %s
	Document ID: %s
	Error: %v
	Result: %+v
	`, pipe["name"], doc["uuid"], err, result)
				return "error_en_extraccion", -1, err
			} */

	/*fs.logger.Success(`
	✅ Structured Extraction Completed
	Pipeline Name: %s
	Document ID: %s
	Page: %d
	Result: %s
	`, pipe["name"], doc["uuid"], page, string(jsonBytes))
			return string(jsonBytes), page, nil */

	return "no_encontrado", -1, nil
}

func (lb *LoadBalancedOpenAIClient) ExtractorInference(ctx context.Context, text, schema string) (*string, error) {
	//lb.logger.Debug("Starting load balanced extraction inference")

	extractor := lb.getNextHealthyClient()
	//lb.logger.Info("Using server %s: %s", extractor.name, extractor.baseURL)

	prompt, err := predictExtract(text, schema)
	if err != nil {
		lb.logger.Error("Failed to prepare extraction prompt: %v", err)
		return nil, err
	}

	resp, err := extractor.client.CreateCompletion(
		ctx,
		openai.CompletionRequest{
			Model:     "extractor",
			MaxTokens: 512,
			Stop:      []string{"<|end-output|>"},
			Prompt:    prompt,
		},
	)

	if err != nil {
		lb.logger.Error("Extraction failed on server %s: %v", extractor.name, err)
		lb.healthMutex.Lock()
		lb.healthChecks[extractor.baseURL] = false
		lb.healthMutex.Unlock()
		return nil, err
	}

	structure := resp.Choices[0].Text
	//lb.logger.Success("Extraction completed successfully on server %s", extractor.name)
	return &structure, nil
}

func (fs *FileService) ApplyAllToBatch(batchID string) error {
	fs.logger.Info("Starting batch processing: %s", batchID)

	// Crear un contexto cancelable para este batch
	ctx, cancel := context.WithCancel(fs.ctx)
	defer cancel()

	documents, err := fs.repo.GetDocumentsFromBatch(batchID)
	if err != nil {
		fs.logger.Error("Failed to get documents from batch: %v", err)
		return fmt.Errorf("error getting documents from batch: %v", err)
	}

	fs.logger.Info("Processing %d documents in batch", len(documents))

	// Canal para errores
	errorChan := make(chan error, len(documents))
	// WaitGroup para esperar que todos los documentos se procesen
	var wg sync.WaitGroup

	// Semáforo para limitar la concurrencia
	sem := make(chan struct{}, fs.config.MaxConcurrentDocuments)

	for _, doc := range documents {
		wg.Add(1)

		go func(doc bson.M) {
			defer wg.Done()

			// Adquirir slot del semáforo
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				return
			}

			docUUID := doc["uuid"].(string)
			fs.logger.Debug("Processing document: %s", docUUID)

			typeDoc := doc["type_document"].(primitive.ObjectID).Hex()
			pipes, err := fs.pipe.GetPipelinesByApplyTo(typeDoc)
			if err != nil {
				fs.logger.Warning("Error getting pipelines for document %s: %v", docUUID, err)
				fs.repo.SetDocumentReviewStatus("error_on_extraction", docUUID)
				errorChan <- err
				return
			}

			pipeArray, ok := pipes["pipelines"].(primitive.A)
			if !ok {
				fs.logger.Error("Invalid pipeline format for document %s", docUUID)
				errorChan <- fmt.Errorf("invalid pipeline format")
				return
			}

			for _, pipe := range pipeArray {
				pipeMap, ok := pipe.(bson.M)
				if !ok {
					fs.logger.Warning("Invalid pipeline item format, skipping")
					continue
				}

				pipelineName := pipeMap["name"].(string)
				//fs.logger.Debug("Applying pipeline '%s' to document %s", pipelineName, docUUID)

				result, page, err := fs.ProcessPipelineToDocument(ctx, pipeMap, doc)
				if err != nil {
					fs.logger.Error("Pipeline '%s' processing failed for document %s: %v",
						pipelineName, docUUID, err)
					continue
				}

				entities := map[string]interface{}{
					"key":     pipeMap["name"],
					"value":   result,
					"page":    page,
					"id_core": pipeMap["id_core"],
				}

				err = fs.repo.AddEntitiesToDocumentInBatch(docUUID, entities)
				if err != nil {
					fs.logger.Error("Failed to add entities to document %s from pipeline '%s': %v\nEntities attempted: %+v",
						docUUID,
						pipelineName,
						err,
						entities)
					continue
				}

				//fs.logger.Success("Successfully processed pipeline '%s' for document %s", pipelineName, docUUID)
			}

			fs.repo.SetDocumentReviewStatus("complete_qa", docUUID)
		}(doc)
	}

	// Esperar a que todos los documentos se procesen
	wg.Wait()

	// Cerrar el canal de errores
	close(errorChan)

	// Procesar errores si los hay
	var errCount int
	for err := range errorChan {
		if err != nil {
			errCount++
			fs.logger.Error("Batch processing error: %v", err)
		}
	}

	if errCount > 0 {
		fs.logger.Warning("Batch completed with %d errors", errCount)
	} else {
		fs.logger.Success("Batch processed successfully")
	}

	return fs.RemoveItem(batchID)
}

// Worker handles the continuous processing of items
type Worker struct {
	fileService *FileService
	stopChan    chan struct{}
	wg          sync.WaitGroup
	logger      *Logger
}

func NewWorker(fileService *FileService) *Worker {
	return &Worker{
		fileService: fileService,
		stopChan:    make(chan struct{}),
		logger:      NewLogger("Worker"),
	}
}

func (w *Worker) Start() {
	w.logger.Info("Starting worker...")
	w.wg.Add(1)
	go w.worker()
}

func (w *Worker) Stop() {
	w.logger.Info("Stopping worker...")
	close(w.stopChan)
	w.wg.Wait()
	w.logger.Success("Worker stopped successfully")
}

func (w *Worker) worker() {
	defer w.wg.Done()
	w.logger.Info("Worker process started")

	for {
		select {
		case <-w.stopChan:
			w.logger.Info("Received stop signal")
			return
		default:
			item, err := w.fileService.GetItemToWork()

			if err == nil {
				if item != "" {
					w.logger.Info("Processing item: %s", item)
					if err := w.fileService.ApplyAllToBatch(item); err != nil {
						w.logger.Critical("Fatal error in batch processing: %v", err)
						return
					}
				}

			} else {
				w.logger.Error(err.Error())
				//w.logger.Debug("No items to process, waiting...")
				time.Sleep(time.Second)
			}
		}
	}
}

func main() {
	mainLogger := NewLogger("Main")
	mainLogger.Info("Starting application...")

	if err := godotenv.Load(); err != nil {
		mainLogger.Warning("No .env file found")
	}

	ctx := context.Background()
	mainLogger.Info("Connecting to MongoDB...")
	mongoURI := os.Getenv("MONGO_URI")
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoURI))
	if err != nil {
		mainLogger.Critical("Failed to connect to MongoDB: %v", err)
		log.Fatal(err)
	}
	defer client.Disconnect(ctx)
	mainLogger.Success("Connected to MongoDB successfully")

	db := client.Database(os.Getenv("MONGO_DB"))
	fileRepo := NewFileRepository(db)
	pipelineRepo := NewPipelineRepository(db)
	searchClient := InitMeiliSearch()

	mainLogger.Info("Initializing services...")
	fileService := NewFileService(fileRepo, pipelineRepo, searchClient)
	worker := NewWorker(fileService)

	// Manejar señales de interrupción
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		mainLogger.Warning("Received signal: %v", sig)
		mainLogger.Info("Initiating graceful shutdown...")
		worker.Stop()
		fileService.Shutdown()
	}()

	mainLogger.Success("Application started successfully")
	worker.Start()
	worker.wg.Wait()
	mainLogger.Info("Application shutdown complete")
}

// Helper functions
func generateSchema(pipe bson.M) string {
	if schema, ok := pipe["schema"].(string); ok && schema != "" {
		return schema
	}

	value, hasValue := pipe["value"].(string)
	key, hasKey := pipe["key"].(string)
	if hasValue && hasKey {
		schemaTemplate := `{
	"%s": {
		"%s": ""
    }
}`
		return fmt.Sprintf(schemaTemplate, key, value)
	}
	return ""
}

func extractValue(data map[string]interface{}) interface{} {
	if len(data) != 1 {
		return data
	}

	for _, v := range data {
		if innerDict, ok := v.(map[string]interface{}); ok && len(innerDict) == 1 {
			for _, innerValue := range innerDict {
				return innerValue
			}
		}
	}
	return data
}
