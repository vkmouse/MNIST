#include "Model.hpp"

Model::Model(const char* saved_model_dir)
{
    graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Buffer* run_opts = NULL;

    const char* tags = "serve"; 

    int ntags = 1;
    session = TF_LoadSessionFromSavedModel(sess_opts, run_opts, saved_model_dir, &tags, ntags, graph, NULL, status);

    size_t pos = 0;
    TF_Operation* oper;

    while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr) {
        std::cout << TF_OperationName(oper) << std::endl;
    }
    TF_DeleteSessionOptions(sess_opts);
}
Model::~Model()
{
    TF_DeleteGraph(graph);
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
}
void Model::run(Tensor input, Tensor& output)
{
    TF_SessionRun(session, NULL, &input.op, &input.val, 1, &output.op, &output.val, 1, NULL, 0, NULL, status);
    output.flag = true;
}
TF_Buffer* Model::read_file(const char* file) 
{
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);                                                                  
    fseek(f, 0, SEEK_SET);  //same as rewind(f);                                            

    void* data = malloc(fsize);                                                             
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBuffer();                                                        
    buf->data = data;
    buf->length = fsize;                                                                    
    // buf->data_deallocator = free_buffer;                                                    
    return buf;
} 