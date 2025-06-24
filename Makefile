# ---------- configuration -------------------------------------------------
STACK_NAME     ?= llm-g5-stack
REGION         ?= eu-central-1
TEMPLATE_FILE  ?= infra/gpu-instance.yaml
KEY_NAME       ?= k8
VPC_ID         ?= vpc-0949ab7a1bd7bdebe
SSH_CIDR       ?= $(shell curl -s https://checkip.amazonaws.com)/32
# --------------------------------------------------------------------------

PARAM_OVERRIDES = \
  KeyName=$(KEY_NAME) \
  VpcId=$(VPC_ID) \
  AllowedSSH=$(SSH_CIDR)

.PHONY: start stop status terminate logs

start:
	@echo "▶ deploying $(STACK_NAME) in $(REGION)…"
	aws cloudformation deploy \
	  --stack-name $(STACK_NAME) \
	  --template-file $(TEMPLATE_FILE) \
	  --region $(REGION) \
	  --capabilities CAPABILITY_IAM \
	  --parameter-overrides $(PARAM_OVERRIDES)
	@echo "⏳ waiting for stack to finish…"
	aws cloudformation wait stack-create-complete \
	  --stack-name $(STACK_NAME) \
	  --region $(REGION)
	@echo "✓ stack ready"
	@$(MAKE) -s status

stop:
	@echo "▶ stopping EC2 instance in $(STACK_NAME)…"
	INSTANCE_ID=$$(aws cloudformation describe-stacks \
	  --stack-name $(STACK_NAME) --region $(REGION) \
	  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
	  --output text); \
	aws ec2 stop-instances --instance-ids $$INSTANCE_ID --region $(REGION); \
	echo "⏳ waiting for stopped state…"; \
	aws ec2 wait instance-stopped --instance-ids $$INSTANCE_ID --region $(REGION); \
	echo "✓ instance $$INSTANCE_ID stopped"

status:
	@aws cloudformation describe-stacks \
	  --stack-name $(STACK_NAME) --region $(REGION) \
	  --query "Stacks[0].Outputs[?OutputKey=='PublicIP' || OutputKey=='InstanceId']" \
	  --output table

terminate:
	@echo "⚠ Deleting the whole stack (instance + EBS root WILL be lost)…"
	aws cloudformation delete-stack --stack-name $(STACK_NAME) --region $(REGION)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK_NAME) --region $(REGION)
	@echo "✓ stack removed"

logs:
	@aws cloudformation describe-stack-events --stack-name $(STACK_NAME) --region $(REGION) --output table
